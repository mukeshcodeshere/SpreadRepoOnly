import pandas as pd
from datetime import datetime
from gcc_sparta_lib import get_mv_data
from datetime import datetime, timedelta
import streamlit as st
import time
import plotly.graph_objects as go
import plotly.express as px
# Today's date
end_date = datetime.today()

# Approximate 15 years ago (15 * 365.25 to account for leap years)
start_date = end_date - timedelta(days=15*365.25)

# Month code mapping
month_code_map = {
    'F': 1,  # Jan
    'G': 2,  # Feb
    'H': 3,  # Mar
    'J': 4,  # Apr
    'K': 5,  # May
    'M': 6,  # Jun
    'N': 7,  # Jul
    'Q': 8,  # Aug
    'U': 9,  # Sep
    'V': 10, # Oct
    'X': 11, # Nov
    'Z': 12  # Dec
}

# Get the current month
current_month = datetime.now().month
current_year = datetime.now().year

# Function to check if an instrument has expired
def check_instrument_expiry_month_only(instruments):
    expired_instruments = []

    for instrument in instruments:
        #print(f"Processing instrument: {instrument}")  # Debugging: Show current instrument

        # Step 1: Check if the instrument code is long enough
        if len(instrument) < 4:
            #print(f"Warning: Instrument code {instrument} is too short to process. Skipping...")
            continue
        
        # Step 2: Drop the last two characters (validate input length)
        shortened_instrument = instrument[:-2]
        #print(f"Shortened instrument (last two characters removed): {shortened_instrument}")  # Debugging: Show shortened instrument

        # Step 3: Take the last character as the month character
        month_char = shortened_instrument[-1]
        #print(f"Extracted month character: {month_char}")  # Debugging: Show extracted month character

        # Step 4: Get the corresponding month number from the month_code_map
        instrument_month = month_code_map.get(month_char, None)

        if instrument_month is None:
            #print(f"Invalid month code: {month_char} in {instrument}. Skipping...")  # Debugging: Show invalid month code
            continue
        
        # Step 5: Compare the current month with the instrument's month
       # print(f"Instrument month: {instrument_month}, Current month: {current_month}")  # Debugging: Show month comparison

        if current_month >= instrument_month:
            expired_instruments.append((instrument, "expired"))
        else:
            expired_instruments.append((instrument, "valid"))
    
    return expired_instruments

def generate_instrument_lists(instrument_expiry_check):
    """
    Generate a new list of instruments based on expiry status and years.
    """
    # Get current year and calculate start_year and end_year
    current_year = datetime.now().year
    start_year = current_year - 10

    # List to store the new instrument lists
    new_instrument_lists = []

    # Process each instrument and create the desired lists
    for instrument, status in instrument_expiry_check:
        # Calculate end_year based on expiration status
        end_year = current_year + 1 if status == "expired" else current_year

        # Generate the list of years
        year_range = [str(year)[2:] for year in range(start_year, end_year + 1)]

        # Remove the last 2 characters from the instrument and concatenate with the years
        instrument_base = instrument[:-2]  # Drop last 2 characters
        new_instrument_list = [instrument_base + year for year in year_range]

        # Add the new instrument list to the result
        new_instrument_lists.append(new_instrument_list)

    # Concatenate all the lists together and remove duplicates
    all_instruments = [instrument for sublist in new_instrument_lists for instrument in sublist]
    unique_instruments = list(set(all_instruments))

    # Sort the unique instruments (optional, but keeps the order consistent)
    unique_instruments.sort()

    return new_instrument_lists,unique_instruments

@st.cache_data
def concatenate_commodity_data_for_unique_instruments(unique_instruments, max_retries=5, retry_delay=5):
    fetched_data = []
    failed_instruments = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_instruments = len(unique_instruments)

    for idx, instrument in enumerate(unique_instruments, start=1):
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                with st.spinner(f"Attempt {attempt}/{max_retries} - Fetching data for {instrument}..."):
                    df_commodity_data = get_mv_data(instrument, start_date, end_date, False)

                if df_commodity_data is not None and not df_commodity_data.empty:
                    df_commodity_data['Instrument'] = instrument
                    fetched_data.append(df_commodity_data)
                    success = True
                    break
                else:
                    print(f" No data returned for {instrument} on attempt {attempt}. Retrying...")
            except Exception as e:
                print(f" Error on attempt {attempt} for {instrument}: {e}")
            
            time.sleep(retry_delay)

        if not success:
            failed_instruments.append(instrument)

        # Update progress bar
        progress = idx / total_instruments
        progress_bar.progress(progress)
        status_text.text(f"Processed {idx}/{total_instruments} instruments")

    if failed_instruments:
        st.error(f"❌ Failed to fetch data for the following instruments after {max_retries} attempts: {', '.join(failed_instruments)}")
    else:
        st.success("✅ All instruments fetched successfully.")

    df_final = pd.concat(fetched_data, ignore_index=True) if fetched_data else pd.DataFrame()
    return df_final

@st.cache_data
def concatenate_commodity_data_for_unique_instruments_mini(unique_instruments, max_retries=5, retry_delay=5):
    fetched_data = []
    failed_instruments = []

    for idx, instrument in enumerate(unique_instruments, start=1):
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                with st.spinner(f"Attempt {attempt}/{max_retries} - Fetching data for {instrument}..."):
                    df_commodity_data = get_mv_data(instrument, start_date, end_date, False)

                if df_commodity_data is not None and not df_commodity_data.empty:
                    df_commodity_data['Instrument'] = instrument
                    fetched_data.append(df_commodity_data)
                    success = True
                    break
                else:
                    print(f" No data returned for {instrument} on attempt {attempt}. Retrying...")
            except Exception as e:
                print(f" Error on attempt {attempt} for {instrument}: {e}")
            
            time.sleep(retry_delay)

        if not success:
            failed_instruments.append(instrument)

    df_final = pd.concat(fetched_data, ignore_index=True) if fetched_data else pd.DataFrame()
    return df_final

def check_instrument_expiry_dict(instruments):
    instrument_status = []
    
    # Get the current year and month for comparison
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    for instrument in instruments:
        if len(instrument) < 4:
            instrument_status.append((instrument, "invalid"))  # Short instruments are invalid
            continue
        
        # Extract the last two digits of the year and convert them to a full year
        instrument_year = 2000 + int(instrument[-2:])
        
        # Extract the month character and map it to a month number
        month_char = instrument[-3]
        instrument_month = month_code_map.get(month_char, None)

        if instrument_month is None:
            instrument_status.append((instrument, "invalid month"))
            continue
        
        # Determine if the instrument is expired or valid
        if (instrument_year < current_year) or (instrument_year == current_year and instrument_month < current_month):
            instrument_status.append((instrument, "expired"))
        else:
            instrument_status.append((instrument, "valid"))
    
    return instrument_status

def plot_seasonality_chart_tab5(df_filtered, meta_A_month_int):
    import plotly.graph_objects as go
    import pandas as pd
    import streamlit as st
    import itertools

    df_expired = df_filtered[df_filtered['ExpiryStatus'] == 'expired']
    df_valid = df_filtered[df_filtered['ExpiryStatus'] == 'valid']

    fig = go.Figure()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Generate unique colors for each instrument
    color_palette = itertools.cycle([
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ])
    instrument_colors = {instrument: next(color_palette) for instrument in df_filtered['Instrument'].unique()}

    # === Plot expired instruments ===
    for (instrument, year), group in df_expired.groupby(['Instrument', 'Year']):
        group = group.sort_values('TradingDayOfYear').tail(252).reset_index(drop=True)
        group = group.copy()
        group.loc[:, 'trading_day_index'] = range(len(group))
        fig.add_trace(go.Scatter(
            x=group['trading_day_index'],
            y=group['Close'],
            mode='lines',
            name=f"{instrument} - {year} (Expired)",
            line=dict(dash='dash', width=2, color=instrument_colors[instrument]),
            opacity=0.7
        ))

    # === Plot valid instruments ===
    if df_valid.empty:
        st.write("No valid instruments found.")
        return

    max_valid_date = df_valid['Date'].max()
    start_year = max_valid_date.year
    start_date = pd.Timestamp(year=start_year, month=meta_A_month_int, day=1)
    previous_year_date = start_date - pd.DateOffset(years=1)

    def date_to_trading_index(date, start):
        days_diff = (date - start).days
        trading_index = days_diff * 5 / 7
        return trading_index

    # Try to get valid data from the initial start_date
    valid_data = df_valid[df_valid['Date'] >= start_date].copy()
    used_start_date = start_date

    if valid_data.empty:
        # If no data, try from the previous year
        valid_data = df_valid[df_valid['Date'] >= previous_year_date].copy()
        used_start_date = previous_year_date

        if valid_data.empty:
            st.write("No valid data after adjusted start date.")
        else:
            for (instrument, year), group in valid_data.groupby(['Instrument', 'Year']):
                group = group.sort_values('Date').copy()
                group.loc[:, 'trading_day_index'] = group['Date'].apply(lambda d: date_to_trading_index(d, used_start_date))
                fig.add_trace(go.Scatter(
                    x=group['trading_day_index'],
                    y=group['Close'],
                    mode='lines',
                    name=f"{instrument} - {year} (Valid)",
                    line=dict(dash='solid', width=3, color=instrument_colors[instrument]),
                    opacity=1
                ))
    else:
        for (instrument, year), group in valid_data.groupby(['Instrument', 'Year']):
            group = group.sort_values('Date').copy()
            group.loc[:, 'trading_day_index'] = group['Date'].apply(lambda d: date_to_trading_index(d, used_start_date))
            fig.add_trace(go.Scatter(
                x=group['trading_day_index'],
                y=group['Close'],
                mode='lines',
                name=f"{instrument} - {year} (Valid)",
                line=dict(dash='solid', width=3, color=instrument_colors[instrument]),
                opacity=1
            ))


    # Month ticks
    month_positions = [i * 21 for i in range(12)]
    month_labels = [month_names[(meta_A_month_int - 1 + i) % 12] for i in range(12)]

    fig.update_layout(
        title=f"📅 Seasonality Chart (Starting from {month_names[meta_A_month_int - 1]})",
        xaxis=dict(title="Month", tickvals=month_positions, ticktext=month_labels),
        yaxis_title="Close Price",
        height=600,
        template="plotly_white",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

# Function to check if a month code is expired or running based on the current month
def check_month_status(month_code_map):
    current_month = datetime.now().month  # Get current month number
    status_dict = {}

    for month_char, month_num in month_code_map.items():
        if month_num <= current_month:
            status_dict[month_char] = 'expired_month'  # Expired if the month is before or equal to current
        else:
            status_dict[month_char] = 'running_month'  # Running if the month is after current

    return status_dict

def plot_spread_seasonality(df_final, base_month_int,current_year):
    # Ensure proper types
    df_final['Date'] = pd.to_datetime(df_final['Date'])

    # Extract year from Base_Instrument (last 2 digits)
    df_final['Year'] = df_final['Base_Instrument'].str.extract(r'(\d{2})$').astype(int) + 2000

    # Sort in descending order by Year and Date to get the most recent data first
    df_final = df_final.sort_values(['Year', 'Date'], ascending=[False, False]).copy()

    # Compute trading day index per year (1 to 252 max)
    df_final['TradingDayOfYear'] = (
        df_final.groupby(['Year'])
        .cumcount() + 1
    )

    # Get the latest year
    latest_year = df_final['Year'].max()
    start_date_for_latest_year = pd.to_datetime(f"{latest_year-1}-{base_month_int:02d}-01")
    # if latest_year > current_year:
    #     # Filter data for the latest year to only include dates starting from the 1st day of base_month_int
    #     start_date_for_latest_year = pd.to_datetime(f"{latest_year-1}-{base_month_int:02d}-01")
    # else:
    #     start_date_for_latest_year = pd.to_datetime(f"{latest_year-1}-{base_month_int:02d}-01")
    df_final.loc[df_final['Year'] == latest_year, 'Date'] = df_final['Date'].where(df_final['Date'] >= start_date_for_latest_year)

    # After applying the date filter, we remove rows with NaT (Not a Time) for the latest year
    df_final = df_final.dropna(subset=['Date'])

    # Select the latest 252 trading days for each year
    df_final = df_final.groupby('Year').head(252)

    # Plot
    fig = go.Figure()

    for year, group in df_final.groupby('Year'):
        # Check if the year is the latest one, and bold it in the legend and graph
        year_label = f"<b>{year}</b>" if year == latest_year else str(year)
        
        line_style = dict(width=4, dash='solid') if year == latest_year else dict(width=2, dash='dot')
        
        fig.add_trace(go.Scatter(
            x=group['TradingDayOfYear'],
            y=group['Spread'],
            mode='lines',
            name=year_label,
            line=line_style,
            hovertext=group['Date'].dt.strftime('%Y-%m-%d'),
            opacity=0.85
        ))

    # X-axis month ticks starting from meta_A_month_int
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_labels = [month_names[(base_month_int - 1 + i) % 12] for i in range(12)]
    month_positions = [i * 21 for i in range(12)]  # ~21 trading days per month

    fig.update_layout(
        title=f"📊 Spread Seasonality Chart (Starting from {month_names[base_month_int - 1]})",
        xaxis=dict(title="Month", tickvals=month_positions, ticktext=month_labels),
        yaxis_title="Spread",
        height=600,
        template="plotly_white",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_kde_distribution(df_final):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import gaussian_kde, norm
    import streamlit as st

    # Extract spread data
    spread_data = df_final['Spread'].dropna().values
    mean_val = np.mean(spread_data)
    median_val = np.median(spread_data)
    std_dev = np.std(spread_data)
    
    # Confidence interval
    z_score = norm.ppf(0.975)
    ci_lower = mean_val - z_score * std_dev / np.sqrt(len(spread_data))
    ci_upper = mean_val + z_score * std_dev / np.sqrt(len(spread_data))

    # KDE
    kde = gaussian_kde(spread_data)
    x_range = np.linspace(min(spread_data), max(spread_data), 1000)
    kde_values = kde(x_range)

    # Subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)

    # KDE + histogram
    fig.add_trace(go.Histogram(
        x=spread_data,
        nbinsx=30,
        histnorm='probability density',
        marker_color='rgba(100, 100, 255, 0.3)',
        opacity=0.6,
        name='Histogram'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_values,
        mode='lines',
        line=dict(color='royalblue', width=3),
        name='KDE'
    ), row=1, col=1)

    # Mean, median, CI
    for val, label, style in [
        (mean_val, f"Mean: {mean_val:.4f}", 'solid'),
        (median_val, f"Median: {median_val:.4f}", 'dash'),
        (ci_lower, f"95% CI Lower", 'dot'),
        (ci_upper, f"95% CI Upper", 'dot')
    ]:
        fig.add_trace(go.Scatter(
            x=[val, val],
            y=[0, max(kde_values) * 1.05],
            mode='lines',
            name=label,
            line=dict(color='gray', dash=style, width=2),
            hoverinfo='name'
        ), row=1, col=1)

    # Boxplot
    fig.add_trace(go.Box(
        x=spread_data,
        boxpoints='outliers',
        marker_color='royalblue',
        name='Boxplot',
        boxmean='sd'
    ), row=2, col=1)

    # Layout
    fig.update_layout(
        title=dict(text="Spread Distribution", x=0.5, font_size=20),
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=80, l=40, r=40, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    fig.update_xaxes(
        title_text="Spread Value",
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False
    )
    fig.update_yaxes(
        title_text="Density",
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False,
        row=1, col=1
    )
    fig.update_yaxes(visible=False, row=2, col=1)

    # Streamlit display
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Distribution Stats"):
        st.metric("Mean", f"{mean_val:.4f}")
        st.metric("Median", f"{median_val:.4f}")
        st.metric("Std Dev", f"{std_dev:.4f}")
        st.metric("95% CI", f"[{ci_lower:.4f}, {ci_upper:.4f}]")

