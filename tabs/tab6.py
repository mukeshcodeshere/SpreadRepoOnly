import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

# Mock data generation functions (replace with your actual data sources)
def generate_mock_price_data(instrument, start_date, end_date):
    """Generate mock price data for demonstration"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(hash(instrument) % 2**32)  # Consistent seed per instrument
    
    # Generate realistic commodity price movements
    returns = np.random.normal(0, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Instrument': instrument
    })

def check_month_status(month_code_map):
    """Check if months are expired or active"""
    status_dict = {}
    for month_code, month_num in month_code_map.items():
        if month_num <= current_month:
            status_dict[month_code] = "expired_month"
        else:
            status_dict[month_code] = "active_month"
    return status_dict

def plot_spread_seasonality(df_final, base_month_int, current_year):
    """Plot spread seasonality analysis"""
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

def plot_kde_distribution(df_final):
    """Plot KDE distribution of spreads"""
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
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Date range selection
        st.subheader("📅 Analysis Period")
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
        
        # Analysis settings
        st.subheader("🔧 Analysis Settings")
        years_back = st.slider("Years of Historical Data", 1, 15, 10)
        
        # Month status info
        st.subheader("📊 Month Status")
        month_status = check_month_status(MONTH_CODE_MAP)
        for month, status in month_status.items():
            color = "🔴" if status == "expired_month" else "🟢"
            st.write(f"{color} {month}: {status.replace('_', ' ').title()}")
    
    # Main content
    st.markdown("## 🔍 Configure Spread Analysis")
    
    # Input method selection
    input_method = st.radio(
        "Input method:",
        ["Quick Selection", "Table Input"],
        help="Quick Selection: Single spread analysis. Table Input: Multiple spreads."
    )
    
    if input_method == "Quick Selection":
        st.markdown("### 🎯 Quick Selection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            root_symbol = st.text_input("Root Symbol", 
                                      value="CL",
                                      help="e.g., CL, GC, ES, NQ")
        
        with col2:
            base_month = st.selectbox("Base Month", 
                                    list(MONTH_CODE_MAP.keys()),
                                    help="The base contract month")
        
        with col3:
            comp_month = st.selectbox("Comparison Month", 
                                    [m for m in MONTH_CODE_MAP.keys() if m != base_month],
                                    help="The comparison contract month")
        
        if st.button("🚀 Analyze Spread", type="primary"):
            if root_symbol and base_month != comp_month:
                spread_configs = [{
                    'root_symbol': root_symbol.upper(),
                    'base_month': base_month,
                    'comparison_month': comp_month,
                    'name': f"{root_symbol.upper()} {comp_month}-{base_month}"
                }]
                process_spreads(spread_configs, start_date, end_date, years_back)
            else:
                st.error("Please ensure root symbol is provided and months are different")
    
    else:
        st.markdown("### 📊 Table Input for Multiple Spreads")
        
        # Default configurations
        default_configs = [
            {"Root Symbol": "CL", "Base Month": "M", "Comparison Month": "Z", "Spread Name": "CL Z-M"},
            {"Root Symbol": "GC", "Base Month": "G", "Comparison Month": "J", "Spread Name": "GC J-G"},
        ]
        
        default_df = pd.DataFrame(default_configs)
        
        spread_df = st.data_editor(
            default_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Root Symbol": st.column_config.TextColumn(
                    "Root Symbol",
                    help="Commodity root symbol (e.g., CL, GC, ES)",
                    max_chars=10
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
                    process_spreads(spread_configs, start_date, end_date, years_back)
            else:
                st.warning("No valid configurations found")

def process_spreads(spread_configs, start_date, end_date, years_back):
    """Process spread configurations and generate analysis"""
    
    for i, config in enumerate(spread_configs):
        st.markdown("---")
        st.markdown(f"### 📈 Analysis {i+1}: {config['name']}")
        
        # Generate instrument combinations
        base_instruments = []
        comp_instruments = []
        
        # Determine year logic based on expiry
        month_status = check_month_status(MONTH_CODE_MAP)
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
            
            # Generate mock data (replace with actual data fetching)
            try:
                df_base = generate_mock_price_data(base_instr, start_date, end_date)
                df_comp = generate_mock_price_data(comp_instr, start_date, end_date)
                
                if df_base.empty or df_comp.empty:
                    continue
                
                # Merge and calculate spread
                df_base = df_base[['Date', 'Close']].rename(columns={'Close': 'Base_Close'})
                df_comp = df_comp[['Date', 'Close']].rename(columns={'Close': 'Comp_Close'})
                
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
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df_final):,}")
            with col2:
                st.metric("Mean Spread", f"{df_final['Spread'].mean():.3f}")
            with col3:
                st.metric("Std Deviation", f"{df_final['Spread'].std():.3f}")
            with col4:
                st.metric("Sharpe-like Ratio", f"{df_final['Spread'].mean()/df_final['Spread'].std():.3f}")
            
            # Generate plots
            base_month_int = MONTH_CODE_MAP[config['base_month']]
            plot_spread_seasonality(df_final, base_month_int, current_year)
            plot_kde_distribution(df_final)
            
            # Data preview
            with st.expander(f"📊 Detailed Data for {config['name']}", expanded=False):
                st.dataframe(df_final, use_container_width=True)
                
                # Additional statistics
                st.subheader("📈 Statistical Summary")
                summary_stats = df_final['Spread'].describe()
                st.dataframe(summary_stats.to_frame().T, use_container_width=True)
                
                # Correlation analysis if multiple instruments
                if len(spread_dfs) > 1:
                    st.subheader("📊 Year-over-Year Correlation")
                    yearly_spreads = df_final.pivot_table(
                        values='Spread', 
                        index=pd.to_datetime(df_final['Date']).dt.dayofyear,
                        columns=pd.to_datetime(df_final['Date']).dt.year,
                        aggfunc='mean'
                    )
                    correlation_matrix = yearly_spreads.corr()
                    st.dataframe(correlation_matrix, use_container_width=True)
        else:
            st.error(f"❌ No data available for {config['name']}")

if __name__ == "__main__":
    main()