import pandas as pd
import streamlit as st
from create_marketview_options import (
    get_expiry_date, get_next_commodity_symbol, append_new_after_old,
    check_expiry
)
from gcc_sparta_lib import get_mv_data

@st.cache_data
def load_commodity_data(symbol, start_date, end_date):
    """
    Load commodity data for a given symbol and date range.
    Gracefully handles connection or data issues.
    """
    try:
        df = get_mv_data(symbol=symbol, start_date=start_date, end_date=end_date)
        if df.empty:
            st.warning(f"No data returned for {symbol}. The data source may be temporarily unavailable.")
        return df
    except Exception as e:
        error_type = type(e).__name__
        st.error(
            f"⚠️ Data loading failed for `{symbol}`.\n\n"
            f"**Error Type:** {error_type}\n"
            f"**Details:** {e}\n\n"
            "This is a common error when the data provider is overloaded or called too frequently. "
            "Please try again in a few minutes."
        )
        return pd.DataFrame()

def process_commodities_data(group_A, group_B, start_date, end_date, available_commodities, group_A_conversion, group_B_conversion):
    def load_group_data(group, group_conversion):
        group_df = pd.DataFrame()
        meta_info = []

        for asset in group:
            symbol = asset["symbol"]
            label = asset["label"]
            weight = asset["weight"]
            conversion = group_conversion.get(symbol, 1)  # Default to 1 if no conversion is provided
            name = available_commodities.get(symbol, symbol)
            
            data = load_commodity_data(symbol, start_date, end_date)

            if data.empty:
                continue

            # Expiry logic
            contract_month, contract_year = symbol[-3:], "20" + symbol[-5:-3]
            data = check_expiry(data, contract_month, contract_year, symbol)
            # Appending logic - not going to do append because only needed during seaso
            # if not data.empty and 'expired' in data.columns and data.expired.unique() == 1:
            #     next_symbol = get_next_commodity_symbol(symbol)
            #     next_data = load_commodity_data(next_symbol, start_date, end_date)
            #     if not next_data.empty:
            #         next_data["Symbol"] = next_symbol
            #         data = append_new_after_old(data, next_data)
            #         data["expired"] = data["expired"].fillna(0)
            #         st.info(f"Appended expired data for {label} with {next_symbol}")

            if not data.empty:
                data = data[["Date", "Close"]].copy()
                data.set_index("Date", inplace=True)
                data.rename(columns={"Close": label}, inplace=True)
                data[label] = data[label] * weight * conversion  # Apply conversion here
                meta_info.append((symbol, label,contract_month))

                if group_df.empty:
                    group_df = data[[label]]
                else:
                    group_df = pd.concat([group_df, data[[label]]], axis=1)

        if not group_df.empty:
            group_df["Group_Mean"] = group_df.mean(axis=1)  # Group total is the mean of individual assets

        return group_df, meta_info

    # Load data for both groups
    df_A, meta_A = load_group_data(group_A, group_A_conversion)
    df_B, meta_B = load_group_data(group_B, group_B_conversion)

    if df_A.empty or df_B.empty:
        st.error("Failed to load data for one or both groups.")
        return pd.DataFrame(), "Group A", "Group B"

    merged = pd.DataFrame(index=df_A.index.union(df_B.index)).sort_index()

    # Add all individual assets to the merged dataframe
    for asset_symbol, asset_label,contract_month in meta_A:
        merged[asset_label] = df_A[asset_label]
    
    for asset_symbol, asset_label,contract_month in meta_B:
        merged[asset_label] = df_B[asset_label]
    
    # Add group totals
    merged["Group A"] = df_A["Group_Mean"]
    merged["Group B"] = df_B["Group_Mean"]
    
    # Clean up missing values
    merged.dropna(inplace=True)
    
    # Calculate spread metrics
    #merged["Spread"] = merged["Group B"] - merged["Group A"]
    merged["Spread"] = merged["Group A"] + merged["Group B"]
    #merged["Spread_Ratio"] = merged["Group B"] / merged["Group A"]

    # Add date as column (keeping the date index)
    merged["Date"] = merged.index

    # Add extra date features
    merged["Year"] = merged.index.year
    merged["Month"] = merged.index.month
    merged["MonthName"] = merged.index.strftime('%b')
    merged["DayOfWeek"] = merged.index.dayofweek
    merged["DayName"] = merged.index.strftime('%a')

    # Calculate returns for group totals and spread
    merged["Return_A"] = merged["Group A"].pct_change()
    merged["Return_B"] = merged["Group B"].pct_change()
    merged["Return_Spread"] = merged["Spread"].pct_change()

    return merged, "Group A", "Group B",meta_A,meta_B

