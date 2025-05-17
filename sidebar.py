import streamlit as st
from datetime import datetime, date
import pandas as pd
from datetime import datetime

current_year = int(datetime.now().year)

def load_presets_from_csv():
    df = pd.read_csv("PriceAnalyzerIn.csv")
    presets = []
    for _, row in df.iterrows():
        presets.append({
            'name': row['Name'],
            'tickers': eval(row['tickerList']),
            'months': eval(row['contractMonthsList']),
            'weights': eval(row['weightsList']),
            'conversions': eval(row['convList']),
            'description': row['desc'],
            'group': row['group'],
            'region': row['region'],
            'months_code': ''.join(eval(row['contractMonthsList']))
        })
    return presets

def show_sidebar(commodity_categories):
    with st.sidebar:
        input_mode = st.radio("Choose Input Mode", ["Preset", "Manual"])
        presets = load_presets_from_csv()
        selected_preset = None

        if input_mode == "Preset":
            st.markdown("### 📚 Presets")

            groups = sorted(set(p['group'] for p in presets))
            selected_group = st.selectbox("Select Group", groups)

            filtered_by_group = [p for p in presets if p['group'] == selected_group]

            regions = sorted(set(p['region'] for p in filtered_by_group))
            selected_region = st.selectbox("Select Region", regions)

            filtered_by_region = [p for p in filtered_by_group if p['region'] == selected_region]

            # Extract and format month codes
            month_codes = sorted(set(p['months_code'] for p in filtered_by_region))
            month_codes_clean = [str(code) for code in month_codes if code]
            month_display = {code: f"Contract: {code}" for code in month_codes_clean}

            selected_month_code = st.selectbox(
                "Select Contract Month-Month",
                options=month_codes_clean,
                format_func=lambda x: month_display.get(x, x)
            )

            # Filter presets based on selected month
            filtered_by_month = [p for p in filtered_by_region if p['months_code'] == selected_month_code]

            # Spread selection
            descriptions = [p['description'] for p in filtered_by_month]
            selected_desc = st.selectbox("Select Spread", descriptions)

            # 💥 FIXED: Select full preset, not just description
            selected_preset = next((p for p in filtered_by_month if p['description'] == selected_desc), None)

            st.markdown("### 📅 Date Range")
            col1, col2 = st.columns(2)
            with col1:
                start_date_input = st.date_input("Start date", value=date(current_year-10, 1, 1), key="start_date")
            with col2:
                end_date_input = st.date_input("End date", value=date(current_year, 12, 31), key="end_date")

            start_date = datetime.combine(start_date_input, datetime.min.time())
            end_date = datetime.combine(end_date_input, datetime.min.time())

            if start_date > end_date:
                st.error("❌ Start date must be before end date.")
                st.stop()

            st.markdown("### ⚙️ Advanced Settings")
            rolling_window = st.slider("Rolling Window Size", min_value=5, max_value=100, value=20, step=1)
            var_confidence = st.slider("VaR Confidence Level (%)", min_value=90, max_value=99, value=95, step=1)

            return [], [], start_date, end_date, rolling_window, var_confidence, 500, selected_preset, presets

        else:
            st.markdown("### 🔍 Select Instruments (Optional)")

            def select_multiple_commodities(group_name, default_weight):
                with st.expander(f"⚙️ Configure {group_name}", expanded=False):
                    category = st.selectbox(
                        f"{group_name}: Exchange / Group",
                        sorted(commodity_categories.keys()),
                        key=f"{group_name}_category"
                    )
                    filtered_items = sorted(commodity_categories[category], key=lambda x: x[1])
                    symbol_options = [f"{desc} ({symbol})" for symbol, desc in filtered_items]
                    selected_labels = st.multiselect(
                        f"{group_name}: Select Instruments",
                        symbol_options,
                        key=f"{group_name}_symbols"
                    )
                    group = []
                    if selected_labels:
                        st.markdown("##### ⚖️ Weights & ⚙️ Conversions")
                    for label in selected_labels:
                        for symbol, desc in filtered_items:
                            if f"{desc} ({symbol})" == label:
                                full_label = f"[{category}] {desc} ({symbol})"
                                cols = st.columns([2, 1, 1])
                                with cols[0]:
                                    st.markdown(f"**{desc} ({symbol})**")
                                with cols[1]:
                                    weight = st.number_input(
                                        "Weight", min_value=-10.0, value=default_weight, step=0.1,
                                        key=f"{group_name}_{symbol}_weight"
                                    )
                                with cols[2]:
                                    conversion = st.number_input(
                                        "Conversion", min_value=-10.0, value=1.0, step=0.1,
                                        key=f"{group_name}_{symbol}_conversion"
                                    )
                                group.append({
                                    "label": full_label,
                                    "symbol": symbol,
                                    "weight": weight,
                                    "conversion": conversion
                                })
                    return group

            group_A = select_multiple_commodities("Group A", default_weight=1.0)
            group_B = select_multiple_commodities("Group B", default_weight=-1.0)

            st.markdown("### 🧾 Add Instruments via Table")
            st.info("Add instruments below. Use the 'Group' column to assign to Group A or B. Delete rows to remove.")

            default_df = pd.DataFrame(columns=["Symbol", "Weight", "Conversion", "Group"])
            instrument_df = st.data_editor(
                default_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Group": st.column_config.SelectboxColumn("Group", options=["Group A", "Group B"]),
                    "Weight": st.column_config.NumberColumn("Weight", default=1.0, min_value=-10.0),
                    "Conversion": st.column_config.NumberColumn("Conversion", default=1.0, min_value=-10.0),
                }
            )

            for idx in instrument_df.index:
                if pd.isna(instrument_df.at[idx, "Symbol"]):
                    continue
                symbol = str(instrument_df.at[idx, "Symbol"])
                group = instrument_df.at[idx, "Group"]

                weight = instrument_df.at[idx, "Weight"]
                if pd.isna(weight):
                    weight = 1.0 if group == "Group A" else -1.0

                conversion = instrument_df.at[idx, "Conversion"]
                if pd.isna(conversion):
                    conversion = 1.0

                entry = {
                    "label": f"[Manual] {symbol}",
                    "symbol": symbol,
                    "weight": float(weight),
                    "conversion": float(conversion)
                }
                if group == "Group A":
                    group_A.append(entry)
                else:
                    group_B.append(entry)

            st.markdown("### 📅 Date Range")
            col1, col2 = st.columns(2)
            with col1:
                start_date_input = st.date_input("Start date", value=date(2010, 1, 1), key="start_date")
            with col2:
                end_date_input = st.date_input("End date", value=date(2025, 12, 31), key="end_date")

            start_date = datetime.combine(start_date_input, datetime.min.time())
            end_date = datetime.combine(end_date_input, datetime.min.time())

            if start_date > end_date:
                st.error("❌ Start date must be before end date.")
                st.stop()

            st.markdown("### ⚙️ Advanced Settings")
            rolling_window = st.slider("Rolling Window Size", min_value=5, max_value=100, value=20, step=1)
            var_confidence = st.slider("VaR Confidence Level (%)", min_value=90, max_value=99, value=95, step=1)

            return group_A, group_B, start_date, end_date, rolling_window, var_confidence, 500, selected_preset, presets

COLORS = {
    "commodity1": "#1f77b4",
    "commodity2": "#2ca02c",
    "spread": "#ff7f0e",
    "background": "#0e1117",
    "text": "#ffffff",
    "grid": "#2c2c2c",
    "highlight": "#e41a1c",
    "appended_data": "#9467bd",
    "latest_data": "#d62728",
    "historical_data": "#7f7f7f",
    "year_colors": {
        2020: "#17becf",
        2021: "#bcbd22",
        2022: "#8c564b",
        2023: "#e377c2",
        2024: "#aec7e8",
        2025: "#98df8a",
        2026: "#ffbb78",
        2027: "#f7b6d2",
        2028: "#c5b0d5",
        2029: "#c49c94",
        2030: "#dbdb8d",
    }
}
