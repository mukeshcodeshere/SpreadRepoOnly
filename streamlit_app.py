import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from scipy import stats
from create_marketview_options import get_available_commodities, categorize_commodities, parse_contract_symbol, check_expiry, append_new_after_old, get_next_commodity_symbol, get_expiry_date
from gcc_sparta_lib import get_mv_data
from tabs.tab6 import render_tab6
from sidebar import show_sidebar, COLORS
from data_engineering import process_commodities_data,load_commodity_data

current_date = datetime.now()
current_month = current_date.month
current_year = current_date.year

# Futures month code mapping (standard convention)
futures_month_map = {
    'F': 1,   # January
    'G': 2,   # February
    'H': 3,   # March
    'J': 4,   # April
    'K': 5,   # May
    'M': 6,   # June
    'N': 7,   # July
    'Q': 8,   # August
    'U': 9,   # September
    'V': 10,  # October
    'X': 11,  # November
    'Z': 12   # December
}

# Set Streamlit page configuration
st.set_page_config(
    page_title="Commodity Market Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom theme
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'axes.facecolor': COLORS["background"],
    'figure.facecolor': COLORS["background"],
    'text.color': COLORS["text"],
    'axes.labelcolor': COLORS["text"],
    'xtick.color': COLORS["text"],
    'ytick.color': COLORS["text"],
    'grid.color': COLORS["grid"],
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

# Styling for Streamlit
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
.section-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #ffffff;
    border-left: 4px solid #2a5298;
    padding-left: 10px;
    margin-bottom: 1rem;
}
.data-info {
    background-color: rgba(42, 82, 152, 0.2);
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
.tooltip {
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted white;
}
.tooltip .tooltiptext {
    visibility: hidden;
    width: 120px;
    background-color: #555;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -60px;
    opacity: 0;
    transition: opacity 0.3s;
}
.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>
<div class="big-title">📊 Project Spartan </div>
""", unsafe_allow_html=True)

# Get the commodities
AVAILABLE_COMMODITIES = get_available_commodities()
commodity_categories = categorize_commodities(AVAILABLE_COMMODITIES)

# Sidebar UI
# In main.py, after getting sidebar inputs:
group_A, group_B, start_date, end_date, rolling_window, var_confidence, chart_height, selected_preset,presets = show_sidebar(commodity_categories)

# Add logic to handle presets
if selected_preset:
    # Find the selected preset data
    preset = selected_preset

    # Clear existing groups
    group_A.clear()
    group_B.clear()

    # Populate Group A and B from preset
    for i, (ticker, month_future, weight, conversion) in enumerate(zip(
            preset['tickers'], 
            preset['months'], 
            preset['weights'], 
            preset['conversions'])):

        # Convert futures month code to integer
        month_int = futures_month_map.get(month_future.upper())
        if not month_int:
            raise ValueError(f"Invalid futures month code: {month_future}")

        # Determine year suffix
        symbol_year = (current_year + 1 if month_int <= current_month else current_year) % 100

        target_group = group_A if i == 0 else group_B
        target_group.append({
            "label": f"[PRESET] {ticker} {month_future}",
            "symbol": f"{ticker}{month_future}{symbol_year}",  # zero-padded year
            "weight": weight,
            "conversion": conversion
        })

# Process the data for multiple assets in each group
merged_data, group_A_name, group_B_name,meta_A,meta_B = process_commodities_data(
    group_A, group_B, start_date, end_date, AVAILABLE_COMMODITIES, 
    {g["symbol"]: g["conversion"] for g in group_A}, 
    {g["symbol"]: g["conversion"] for g in group_B},  # Passing conversion values
)
# Benchmark is Group A's first instrument
meta_A_month_letter = meta_A[0][2][0]
meta_A_month_int = futures_month_map.get(meta_A_month_letter.upper())
list_of_input_instruments = [item[0] for item in (meta_A + meta_B)]

# Handle the merged data
if not merged_data.empty:
    st.info("Data is available for analysis.")
    with st.expander("Click to view merged data", expanded=False):
        st.dataframe(merged_data)
else:
    st.warning("No data available for the selected commodities.")

# Create tabs for different analyses
tab6, = st.tabs(["🌦️ Raul-Seasonality-Spreads"])

with tab6:
    render_tab6(list_of_input_instruments)

# with tab1:
#     render_tab1(
#         merged_data=merged_data,
#         group_A_name=group_A_name,
#         group_B_name=group_B_name,
#         chart_height=chart_height,
#         COLORS=COLORS
#     )

# with tab2:
#     render_tab2(tab2, merged_data, rolling_window, chart_height, COLORS, group_A_name, group_B_name)

# with tab3:
#     render_tab3(merged_data, [group_A_name, group_B_name],meta_A_month_int,list_of_input_instruments)

# with tab4:
#     render_tab4(
#         merged_data=merged_data,
#         group_A_name=group_A_name,
#         group_B_name=group_B_name,
#         var_confidence=var_confidence,
#         COLORS=COLORS
#     )

# with tab5:
#     render_tab5(merged_data, [group_A_name, group_B_name],meta_A_month_int,list_of_input_instruments)

# with tab6:
#     render_tab6(list_of_input_instruments)