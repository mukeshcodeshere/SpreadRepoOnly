import streamlit as st
import pandas as pd
from datetime import datetime
from data_engineering_tab6 import generate_instrument_lists, check_instrument_expiry_month_only, check_month_status,concatenate_commodity_data_for_unique_instruments_mini,plot_spread_seasonality,plot_kde_distribution

# Month character code mapping
month_code_map = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}

# Get current year and month (for expiry comparison)
current_year = datetime.now().year
current_month = datetime.now().month

def render_tab6(list_of_input_instruments):
    st.markdown("## 📈 Spread Analysis")

    # 1. Expiry status check
    month_status_dict = check_month_status(month_code_map)
    
    # 2. Generate unique instruments and root symbols
    instrument_expiry_check = check_instrument_expiry_month_only(list_of_input_instruments)
    _, unique_instruments = generate_instrument_lists(instrument_expiry_check)
    root_symbols = sorted(set(inst[:-3] for inst in unique_instruments))

    # UI Section: Selection for Root Symbol and Months
    st.markdown("---")
    st.subheader("🔍 Select Instrument and Months [Spread = Compared Instrument - Base Instrument]")

    # Use expanders to organize sections and reduce clutter
    with st.expander("Select Root Symbol", expanded=True):
        selected_root = st.selectbox("Choose Root Symbol:", root_symbols, key="root_symbol")
        custom_root = st.text_input("Or enter a custom root symbol:", key="custom_root").strip()

    # Final root selection logic
    selected_root = custom_root if custom_root else selected_root
    
    with st.expander("Select Months for Comparison", expanded=True):
        # Select base month
        selected_base_month = st.selectbox(
            "Select Base Month Code:", 
            list(month_code_map.keys()), 
            key="base_month"
        )

        # Create a new list excluding the selected base month
        comparison_month_options = [month for month in month_code_map.keys() if month != selected_base_month]

        # Select comparison month from the filtered list
        selected_comparison_month = st.selectbox(
            "Select Comparison Month Code:", 
            comparison_month_options, 
            key="comparison_month"
        )

        # Adding expiry status for each month from month_status_dict
        base_expiry_status = month_status_dict.get(selected_base_month, "Unknown")        
        # Determine the base year
        if base_expiry_status == "expired_month":
            base_year = current_year + 1  # If expired, base year is next year
        else:
            base_year = current_year  # If not expired, base year is the current year
        
        # Subtract 2000 from both base and comparison years
        base_year_minus_2k = base_year - 2000
        comparison_year_minus_2k = current_year - 2000  # Comparison year is always the current year - 2000
    
    # Display selected base and comparison months with their expiry status
    if selected_root and selected_base_month and selected_comparison_month:
        # Ensure base month and comparison month are different
        if selected_base_month != selected_comparison_month:
            base_instr = f"{selected_root}{selected_base_month}{base_year_minus_2k}"  # Replace with calculated base year
            comp_instr = f"{selected_root}{selected_comparison_month}{comparison_year_minus_2k}"
            # Initialize lists to store instruments
            base_instr_list = []
            comp_instr_list = []

            # Generate the instruments for the base month and comparison month
            for year_offset in range(-10, 1):  # Range from base_year_minus_2k - 10 to base_year_minus_2k
                base_year = base_year_minus_2k + year_offset
                comparison_year = base_year_minus_2k + year_offset

                # Generate the base_instr and comp_instr for each year offset
                base_instr = f"{selected_root}{selected_base_month}{base_year}"
                comp_instr = f"{selected_root}{selected_comparison_month}{comparison_year}"

                # Append to respective lists
                base_instr_list.append(base_instr)
                comp_instr_list.append(comp_instr)

            spread_dfs = []

            # Iterate over instrument pairs from the same year
            for base_instr, comp_instr in zip(base_instr_list, comp_instr_list):
                df_base = concatenate_commodity_data_for_unique_instruments_mini([base_instr], max_retries=3, retry_delay=3)
                df_comp = concatenate_commodity_data_for_unique_instruments_mini([comp_instr], max_retries=3, retry_delay=3)

                # Skip if either is None or empty
                if df_base is None or df_comp is None or df_base.empty or df_comp.empty:
                    continue

                # Keep only relevant columns and drop missing dates
                df_base = df_base[['Date', 'Close']].dropna(subset=['Date']).rename(columns={'Close': 'Base_Close'})
                df_comp = df_comp[['Date', 'Close']].dropna(subset=['Date']).rename(columns={'Close': 'Comp_Close'})

                # Merge on Date
                df_merged = pd.merge(df_comp, df_base, on='Date', how='inner')
                df_merged['Spread'] = df_merged['Comp_Close'] - df_merged['Base_Close']
                
                # Track which instruments the spread came from
                df_merged['Base_Instrument'] = base_instr
                df_merged['Comp_Instrument'] = comp_instr

                spread_dfs.append(df_merged)

            # Combine all into one final DataFrame
            df_final = pd.concat(spread_dfs, ignore_index=True)

            # Reorder columns
            df_final = df_final[['Date', 'Base_Instrument', 'Comp_Instrument', 'Base_Close', 'Comp_Close', 'Spread']]

            # Example base_month_int = 5 (May), or whatever user selected earlier
            base_month_int = month_code_map[selected_base_month]
            plot_spread_seasonality(df_final, base_month_int,current_year)

            # Plot KDE Distribution of Spread
            plot_kde_distribution(df_final)

            # DataFrame preview
            with st.expander("📊 Preview Spread Data",expanded=False):
                st.dataframe(df_final)

        else:
            st.warning("Base Month and Comparison Month should be different. Please select different months.")

    else:
        st.warning("Please select both the Base and Comparison months.")
