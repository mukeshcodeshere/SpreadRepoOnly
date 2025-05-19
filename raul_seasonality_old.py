#raul_seasonality.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import real data functions from tab6's data engineering module
try:
    from raul_seasonality_engineering import (
        generate_instrument_lists, 
        check_instrument_expiry_month_only, 
        check_month_status,
        concatenate_commodity_data_for_unique_instruments_mini,
        plot_spread_seasonality,
        plot_kde_distribution
    )
    REAL_DATA_AVAILABLE = True
except ImportError:
    st.error("❌ Cannot import data_engineering_tab6 module. Please ensure it's in the same directory.")
    REAL_DATA_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Spread Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
.big-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: #ffffff;
    text-align: center;
    margin-bottom: 1.5rem;
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 1rem;
    border-radius: 10px;
}
.metric-container {
    background-color: rgba(42, 82, 152, 0.2);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
.data-source-indicator {
    position: fixed;
    top: 10px;
    right: 10px;
    background-color: #28a745;
    color: white;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 0.8rem;
    z-index: 1000;
}
</style>
""", unsafe_allow_html=True)

# Month character code mapping
MONTH_CODE_MAP = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}

# Get current year and month
current_year = datetime.now().year
current_month = datetime.now().month

# Add data source indicator
if REAL_DATA_AVAILABLE:
    st.markdown('<div class="data-source-indicator">🟢 REAL DATA</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="data-source-indicator" style="background-color: #dc3545;">🔴 MOCK DATA</div>', unsafe_allow_html=True)

# Add this to your sidebar configuration section:
def add_month_filter_controls():
    """Add month filter controls to the sidebar"""
    st.subheader("📅 Month Filter")
    
    enable_filter = st.checkbox("Enable month filtering", help="Filter data to specific months relative to the base month")
    
    if enable_filter:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Two-value slider for month range
        month_range = st.slider(
            "Month range (relative to base month)",
            min_value=1,
            max_value=12,
            value=(1, 12),
            help="Select months 1-12 relative to the base contract month"
        )
        
        # Show explanation
        #st.info(f"Months {month_range[0]} to {month_range[1]} relative to base month will be included")
        st.info(f"Months {month_range[0]} to {month_range[1]}")
        # Advanced: Show actual month names if base month is known
        # This would require passing the base month from the main configuration
        
        return month_range
    else:
        return None

def fetch_real_data(instrument, max_retries=3, retry_delay=3):
    """Fetch real data for an instrument"""
    if not REAL_DATA_AVAILABLE:
        st.warning(f"Real data not available for {instrument}, using mock data")
        return None
    
    try:
        df = concatenate_commodity_data_for_unique_instruments_mini(
            [instrument], 
            max_retries=max_retries, 
            retry_delay=retry_delay
        )
        return df
    except Exception as e:
        st.warning(f"Error fetching real data for {instrument}: {str(e)}")
        return None

def get_available_root_symbols(list_of_input_instruments=None):
    """Get available root symbols from real data or return common ones"""
    if REAL_DATA_AVAILABLE and list_of_input_instruments:
        try:
            instrument_expiry_check = check_instrument_expiry_month_only(list_of_input_instruments)
            _, unique_instruments = generate_instrument_lists(instrument_expiry_check)
            root_symbols = sorted(set(inst[:-3] for inst in unique_instruments))
            return root_symbols
        except:
            pass
    
    # Fallback to common root symbols
    return ['/CL', '/HO', '/GCL' '/GC', '/SI', '/ES', '/NQ', '/ZC', '/ZS', '/ZW', '/NG']

def plot_spread_seasonality_enhanced(df_final, base_month_int, current_year):
    """Enhanced spread seasonality analysis (using the original from tab6 if available)"""
    if REAL_DATA_AVAILABLE:
        try:
            # Use the original function from tab6
            plot_spread_seasonality(df_final, base_month_int, current_year)
            return
        except:
            pass
    
    # Fallback implementation
    if df_final.empty:
        st.error("No data available for seasonality plot")
        return
    
    # Extract month from Date
    df_final['Month'] = pd.to_datetime(df_final['Date']).dt.month
    df_final['Year'] = pd.to_datetime(df_final['Date']).dt.year
    
    # Create monthly spread statistics
    monthly_stats = df_final.groupby('Month')['Spread'].agg(['mean', 'std', 'count']).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Average spread by month
    ax1.bar(monthly_stats['Month'], monthly_stats['mean'], 
            color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Spread')
    ax1.set_title('Average Spread by Month')
    ax1.grid(True, alpha=0.3)
    
    # Add month labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_names)
    
    # Plot 2: Spread by year and month (heatmap style)
    yearly_monthly = df_final.pivot_table(values='Spread', index='Year', columns='Month', aggfunc='mean')
    
    # Create heatmap
    im = ax2.imshow(yearly_monthly.values, cmap='RdBu_r', aspect='auto')
    ax2.set_xticks(range(len(yearly_monthly.columns)))
    ax2.set_xticklabels([month_names[i-1] for i in yearly_monthly.columns])
    ax2.set_yticks(range(len(yearly_monthly.index)))
    ax2.set_yticklabels(yearly_monthly.index)
    ax2.set_title('Spread Heatmap by Year and Month')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Average Spread')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Fallback implementation
    if df_final.empty:
        st.error("No data available for distribution plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Histogram and KDE
    ax1.hist(df_final['Spread'], bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    
    # Add KDE curve
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(df_final['Spread'])
    x_range = np.linspace(df_final['Spread'].min(), df_final['Spread'].max(), 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    ax1.axvline(df_final['Spread'].mean(), color='green', linestyle='--', label=f'Mean: {df_final["Spread"].mean():.2f}')
    ax1.set_xlabel('Spread Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Spread Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot by year
    df_final['Year'] = pd.to_datetime(df_final['Date']).dt.year
    years = sorted(df_final['Year'].unique())
    spread_by_year = [df_final[df_final['Year'] == year]['Spread'].values for year in years]
    
    ax2.boxplot(spread_by_year, labels=years)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Spread Value')
    ax2.set_title('Spread Distribution by Year')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.markdown('<div class="big-title">📈 Commodity Spread Analysis Tool</div>', unsafe_allow_html=True)
    
    # Display data source status
    if REAL_DATA_AVAILABLE:
        st.success("✅ Using real market data from data_engineering_tab6")
    else:
        st.warning("⚠️ Real data module not available. Using mock data for demonstration.")
    
    # Initialize list_of_input_instruments for real data mode
    list_of_input_instruments = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection (if real data available)
        if REAL_DATA_AVAILABLE:
            st.subheader("📊 Data Source")
            use_instrument_list = st.checkbox("Use existing instrument list", value=False)
            if use_instrument_list:
                # This would need to be passed from the main application
                # For now, we'll use a default list
                list_of_input_instruments = ['CLZ24', 'CLF25', 'GCZ24', 'GCF25']  # Example
                st.info(f"Using {len(list_of_input_instruments)} instruments from existing list")
        
        # Date range selection (for mock data)
        if not REAL_DATA_AVAILABLE:
            st.subheader("📅 Analysis Period (Mock Data)")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", 
                                         value=datetime(current_year-5, 1, 1),
                                         max_value=datetime.now())
            with col2:
                end_date = st.date_input("End Date", 
                                       value=datetime.now(),
                                       max_value=datetime.now())
            
            if start_date >= end_date:
                st.error("Start date must be before end date")
                return
        else:
            start_date = None
            end_date = None
        
        # Analysis settings
        st.subheader("🔧 Analysis Settings")
        years_back = st.slider("Years of Historical Data", 1, 15, 10)
        if REAL_DATA_AVAILABLE:
            max_retries = st.slider("Max Retries for Data Fetch", 1, 5, 3)
            retry_delay = st.slider("Retry Delay (seconds)", 1, 10, 3)
        else:
            max_retries = 3
            retry_delay = 3
        
        # Month status info
        st.subheader("📊 Month Status")
        if REAL_DATA_AVAILABLE:
            month_status = check_month_status(MONTH_CODE_MAP)
        else:
            # Fallback month status calculation
            month_status = {}
            for month_code, month_num in MONTH_CODE_MAP.items():
                if month_num <= current_month:
                    month_status[month_code] = "expired_month"
                else:
                    month_status[month_code] = "active_month"
        
        for month, status in month_status.items():
            color = "🔴" if status == "expired_month" else "🟢"
            st.write(f"{color} {month}: {status.replace('_', ' ').title()}")
        
        month_filter = add_month_filter_controls()

    
    # Main content
    st.markdown("## 🔍 Configure Spread Analysis")
    
    # Input method selection
    input_method = st.radio(
        "Input method:",
        ["Single Spread Selection", "Table Input"],
        help="Quick Selection: Single spread analysis. Table Input: Multiple spreads."
    )
    
    # Get available root symbols
    available_root_symbols = get_available_root_symbols(list_of_input_instruments)
    
    if input_method == "Single Spread Selection":
        st.markdown("### 🎯 Single Spread Selection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            root_symbol = st.selectbox("Root Symbol", 
                                     available_root_symbols,
                                     help="Available root symbols")
            custom_root = st.text_input("Or enter custom MV sybmol (eg. /GCL):", help="Custom root symbol")
            
        # Use custom root if provided, otherwise use selected
        final_root = custom_root.strip().upper() if custom_root else root_symbol
        
        with col2:
            base_month = st.selectbox("Base Month", 
                                    list(MONTH_CODE_MAP.keys()),
                                    help="The base contract month")
        
        with col3:
            comp_month = st.selectbox("Comparison Month", 
                                    [m for m in MONTH_CODE_MAP.keys() if m != base_month],
                                    help="The comparison contract month")
        
        if st.button("🚀 Analyze Spread", type="primary"):
            if final_root and base_month != comp_month:
                spread_configs = [{
                    'root_symbol': final_root,
                    'base_month': base_month,
                    'comparison_month': comp_month,
                    'name': f"{final_root} {comp_month}-{base_month}"
                }]
                process_spreads(spread_configs, start_date, end_date, years_back, max_retries, retry_delay, month_filter)
            else:
                st.error("Please ensure root symbol is provided and months are different")
    
    else:
        st.markdown("### 📊 Table Input for Multiple Spreads")
        
        # Default configurations
        default_configs = [
            {"Root Symbol": available_root_symbols[0] if available_root_symbols else "CL", 
             "Base Month": "M", "Comparison Month": "Z", 
             "Spread Name": f"{available_root_symbols[0] if available_root_symbols else 'CL'} Z-M"},
            {"Root Symbol": available_root_symbols[1] if len(available_root_symbols) > 1 else "GC", 
             "Base Month": "G", "Comparison Month": "J", 
             "Spread Name": f"{available_root_symbols[1] if len(available_root_symbols) > 1 else 'GC'} J-G"},
        ]
        
        default_df = pd.DataFrame(default_configs)
        
        spread_df = st.data_editor(
            default_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Root Symbol": st.column_config.SelectboxColumn(
                    "Root Symbol",
                    help="Root symbol from available list or custom",
                    options=available_root_symbols + ["Custom"]
                ),
                "Base Month": st.column_config.SelectboxColumn(
                    "Base Month",
                    help="Base contract month",
                    options=list(MONTH_CODE_MAP.keys())
                ),
                "Comparison Month": st.column_config.SelectboxColumn(
                    "Comparison Month",
                    help="Comparison contract month",
                    options=list(MONTH_CODE_MAP.keys())
                ),
                "Spread Name": st.column_config.TextColumn(
                    "Spread Name",
                    help="Custom name for the spread",
                    max_chars=50
                )
            }
        )
        
        if not spread_df.empty:
            # Validate configurations
            spread_configs = []
            for idx, row in spread_df.iterrows():
                if pd.isna(row["Root Symbol"]) or pd.isna(row["Base Month"]) or pd.isna(row["Comparison Month"]):
                    continue
                
                root_symbol = str(row["Root Symbol"]).strip().upper()
                base_month = str(row["Base Month"]).strip()
                comp_month = str(row["Comparison Month"]).strip()
                
                if base_month == comp_month:
                    st.error(f"Row {idx + 1}: Base and Comparison months must be different")
                    continue
                
                spread_name = row["Spread Name"] if pd.notna(row["Spread Name"]) else f"{root_symbol} {comp_month}-{base_month}"
                
                spread_configs.append({
                    'root_symbol': root_symbol,
                    'base_month': base_month,
                    'comparison_month': comp_month,
                    'name': spread_name
                })
            
            if spread_configs:
                st.success(f"✅ {len(spread_configs)} valid spread configuration(s)")
                
                if st.button("🚀 Analyze All Spreads", type="primary"):
                    process_spreads(spread_configs, start_date, end_date, years_back, max_retries, retry_delay, month_filter)
            else:
                st.warning("No valid configurations found")

# Update your process_spreads function to use the filter:
def process_spreads(spread_configs, start_date, end_date, years_back, max_retries, retry_delay, month_filter=None):
    """Process spread configurations and generate analysis using real or mock data"""
    
    for i, config in enumerate(spread_configs):
        st.markdown("---")
        st.markdown(f"### 📈 Analysis {i+1}: {config['name']}")
        
        # Generate instrument combinations
        base_instruments = []
        comp_instruments = []
        
        # Determine year logic based on expiry
        if REAL_DATA_AVAILABLE:
            month_status = check_month_status(MONTH_CODE_MAP)
        else:
            # Fallback logic
            month_status = {}
            for month_code, month_num in MONTH_CODE_MAP.items():
                if month_num <= current_month:
                    month_status[month_code] = "expired_month"
                else:
                    month_status[month_code] = "active_month"
        
        base_expiry = month_status.get(config['base_month'], "Unknown")
        
        # Current year logic
        if base_expiry == "expired_month":
            base_year = current_year + 1
        else:
            base_year = current_year
        
        # Generate historical instruments
        for year_offset in range(-years_back, 1):
            year_2digit = (base_year + year_offset) % 100
            
            base_instr = f"{config['root_symbol']}{config['base_month']}{year_2digit:02d}"
            comp_instr = f"{config['root_symbol']}{config['comparison_month']}{year_2digit:02d}"
            
            base_instruments.append(base_instr)
            comp_instruments.append(comp_instr)
        
        # Show instruments being analyzed
        with st.expander("📋 Instruments Being Analyzed", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Base Instruments:**")
                st.write(base_instruments)
            with col2:
                st.write("**Comparison Instruments:**")
                st.write(comp_instruments)
        
        # Generate and process data
        spread_dfs = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (base_instr, comp_instr) in enumerate(zip(base_instruments, comp_instruments)):
            progress = (idx + 1) / len(base_instruments)
            progress_bar.progress(progress)
            status_text.text(f"Processing {base_instr} vs {comp_instr}...")
            
            try:
                if REAL_DATA_AVAILABLE:
                    # Fetch real data
                    df_base = fetch_real_data(base_instr, max_retries, retry_delay)
                    df_comp = fetch_real_data(comp_instr, max_retries, retry_delay)
                else:
                    # Generate mock data would go here
                    df_base = None
                    df_comp = None
                
                if df_base is None or df_comp is None or df_base.empty or df_comp.empty:
                    continue
                
                # Merge and calculate spread
                df_base = df_base[['Date', 'Close']].dropna(subset=['Date']).rename(columns={'Close': 'Base_Close'})
                df_comp = df_comp[['Date', 'Close']].dropna(subset=['Date']).rename(columns={'Close': 'Comp_Close'})
                
                df_merged = pd.merge(df_comp, df_base, on='Date', how='inner')
                df_merged['Spread'] = df_merged['Comp_Close'] - df_merged['Base_Close']
                df_merged['Base_Instrument'] = base_instr
                df_merged['Comp_Instrument'] = comp_instr
                
                spread_dfs.append(df_merged)
                
            except Exception as e:
                st.warning(f"Could not process {base_instr} vs {comp_instr}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        if spread_dfs:
            # Combine all spread data
            df_final = pd.concat(spread_dfs, ignore_index=True)
            df_final = df_final[['Date', 'Base_Instrument', 'Comp_Instrument', 'Base_Close', 'Comp_Close', 'Spread']]
            
            # Apply month filtering information for display
            filtered_info = ""
            if month_filter:
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                base_month_int = MONTH_CODE_MAP[config['base_month']]
                start_name = month_names[(base_month_int - 1 + month_filter[0] - 1) % 12]
                end_name = month_names[(base_month_int - 1 + month_filter[1] - 1) % 12]
                filtered_info = f" (Filtered: {start_name} to {end_name})"
            
            # Display summary statistics
            st.markdown(f"#### Summary Statistics{filtered_info}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{len(df_final):,}")
            with col2:
                st.metric("Mean Spread", f"{df_final['Spread'].mean():.3f}")
            with col3:
                st.metric("Std Deviation", f"{df_final['Spread'].std():.3f}")
            
            # Generate plots with month filtering
            base_month_int = MONTH_CODE_MAP[config['base_month']]
            
            # Call updated plotting functions with month_filter parameter
            plot_spread_seasonality(df_final, base_month_int, current_year, month_filter)
            plot_kde_distribution(df_final, month_filter)
            
            # Data preview
            with st.expander(f"📊 Detailed Data for {config['name']}", expanded=False):
                # Show filtered data count if filter is applied
                if month_filter and 'MonthFromBase' in df_final.columns:
                    start_month, end_month = month_filter
                    if start_month <= end_month:
                        filtered_df = df_final[
                            (df_final['MonthFromBase'] >= start_month) & 
                            (df_final['MonthFromBase'] <= end_month)
                        ]
                    else:
                        filtered_df = df_final[
                            (df_final['MonthFromBase'] >= start_month) | 
                            (df_final['MonthFromBase'] <= end_month)
                        ]
                    st.info(f"Showing {len(filtered_df):,} records out of {len(df_final):,} total records (filtered)")
                    st.dataframe(filtered_df, use_container_width=True)
                else:
                    st.dataframe(df_final, use_container_width=True)
                
                # Additional statistics
                st.subheader("📈 Statistical Summary")
                summary_stats = df_final['Spread'].describe()
                st.dataframe(summary_stats.to_frame().T, use_container_width=True)
        else:
            st.error(f"❌ No data available for {config['name']}")
if __name__ == "__main__":
    main()