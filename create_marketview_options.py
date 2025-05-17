from datetime import datetime
from collections import defaultdict
import pandas as pd
import pandas_market_calendars as mcal

def get_available_commodities():
    # Base roots and descriptions
    AVAILABLE_COMMODITIES = {
        # NYMEX Futures
        "/GCL": "Light, Sweet Crude Oil",
        # "/GHO": "Heating Oil",
        # "/GRB": "RBOB Unleaded Gasoline",
        # "/GNG": "Natural Gas",
        # "/GPA": "Palladium",
        # "/GPL": "Platinum",
        # "/GWS": "WTI Financial",
        # "/GHP": "Henry Hub Financial",
        # "/GHH": "Henry Hub Last Day Financial",
        # "/GRT": "RBOB Financial",
        # # CME Futures
        # "/LE": "Live Cattle",
        # "/GF": "Feeder Cattle",
        # "/HE": "Lean Hog",
        # # CBOT Agriculture
        # "/ZC": "CBOT Corn",
        # "/ZE": "Denatured Fuel Ethanol",
        # "/ZM": "Soybean Meal",
        # "/ZL": "Soybean Oil",
        # "/ZS": "Soybeans",
        # "/ZB": "30 YR Bonds",
        # "/ZW": "Wheat",
        # "/KE": "Hard Red Winter Wheat",
        # "/EH": "Ethanol Forward Month Swap",
        # # CBOT US Treasury
        # "/ZT": "2 Year T-Notes",
        # "/ZF": "5 Year T-Notes",
        # "/ZN": "10 Year T-Notes",
        # "/ZQ": "30 day Federal Funds",
        # # COMEX
        # "/GGC": "Gold",
        # "/GHG": "Copper",
        # "/GSI": "Silver",
        # # DME
        # "/OQ": "Oman Crude Oil",
        # "/ZG": "Oman Crude Oil Financial",
        # "/SOQ": "Oman Crude Oil Singapore",
        # "/SZG": "Oman Crude Oil Sing Financial",
        # # TOCOM
        # "/CRUD": "Crude Oil (TOCOM)",
        # "/GASO": "Gasoline (TOCOM)",
        # "/GOLD": "Gold (TOCOM)",
        # "/KERO": "Kerosene",
        # "/PALL": "Palladium (TOCOM)",
        # "/PLAT": "Platinum (TOCOM)",
        # "/RSS3": "Rubber",
        # "/SILV": "Silver (TOCOM)",
        # # CME FX
        # "/6A": "Australian Dollar",
        # "/6B": "British Pound",
        # "/6C": "Canadian Dollar",
        # "/6E": "Euro FX",
        # "/6J": "Japanese Yen",
        # "/6N": "New Zealand Dollar",
        # "/6S": "Swiss Franc",
        # "/NOK": "Norwegian Krone",
        # "/SEK": "Swedish Krona",
        # "/E7": "E-mini Euro FX",
        # "/J7": "E-mini Japanese Yen",
        # "/FXD": "FX Index",
        # # ICE Futures Europe (IPE)
        # "/BRN": "Brent Crude – Electronic",
        # "/GAS": "Gasoil – Electronic", #ICEGCMMF26--> /GASM26 (similar items?)
        # "/WBS": "WTI (ICE)",
        # "/UHO": "Heating Oil (ICE)",
        # "/UHU": "RBOB Gasoline (ICE)",
        # "/BPM": "Electricity – Baseload Month",
        # "/PPM": "Electricity – Peak Month",
        # "/GWM": "Natural Gas – Month",
        # "/MID": "Middle East Sour Crude",
        # "/ATW": "Rotterdam Coal",
        # "/AFR": "Richards Bay Coal",
        # "/NCF": "Newcastle Coal",
        # "/ECF": "ECX Carbon Financial Futures",
        # "/CER": "ECX CER Futures",
        # # ICE Futures US
        # "/1sKC": "Coffee (ICE)",
        # "/1sCT": "Cotton (ICE)",
        # "/1sCC": "Cocoa (ICE)",
        # "/1sSB": "Sugar No. 11 (ICE)",
        # "/1sOJ": "Orange Juice (ICE)",
        # "/1sIS": "Soybean (ICE)",
        # "/1sISM": "Soybean Meal (ICE)",
        # "/1sIBO": "Soybean Oil (ICE)"
    }

    # Month code mapping
    MONTH_CODES = {
        'F': 'JAN', 'G': 'FEB', 'H': 'MAR', 'J': 'APR', 'K': 'MAY', 'M': 'JUN',
        'N': 'JUL', 'Q': 'AUG', 'U': 'SEP', 'V': 'OCT', 'X': 'NOV', 'Z': 'DEC'
    }
    MONTHS_3L = list(MONTH_CODES.values())
    current_year = datetime.now().year
    years_to_extend_by = 3
    years = [current_year + i for i in range(years_to_extend_by)]  # Current and next years_to_extend_by years
    year_suffixes = [str(y)[-2:] for y in years]

    # --- Generate Futures, Alias, Rolling, Continuous, and Strip variations ---
    variations = {}

    for root, desc in list(AVAILABLE_COMMODITIES.items()):
        # Standard Futures: /ROOTmyy
        for mcode in MONTH_CODES:
            for yys in year_suffixes:
                symbol = f"{root}{mcode}{yys}"
                variations[symbol] = f"{desc} {MONTH_CODES[mcode].capitalize()} 20{yys}"

        # # Alias: /ROOT[0], /ROOT[1], /ROOT[2]
        # for i in range(years_to_extend_by):
        #     variations[f"{root}[{i}]"] = f"{desc} Alias Month {i+1}"
        #     # Globex Alias: G/ROOT[0], etc.
        #     variations[f"G{root}[{i}]"] = f"{desc} Globex Alias Month {i+1}"

        # # Rolling: /ROOT<0>, /ROOT<1>, /ROOT<2>
        # for i in range(years_to_extend_by):
        #     variations[f"{root}<{i}>"] = f"{desc} Rolling Month {i+1}"
        #     # Globex Rolling: G/ROOT<0>, etc.
        #     variations[f"G{root}<{i}>"] = f"{desc} Globex Rolling Month {i+1}"

        # # Continuous: /ROOT<JAN>, /ROOT<JAN+1>, /ROOT<JAN+2>
        # for mon in MONTHS_3L:
        #     variations[f"{root}<{mon}>"] = f"{desc} Continuous {mon.capitalize()}"
        #     for out in range(1, years_to_extend_by):
        #         variations[f"{root}<{mon}+{out}>"] = f"{desc} Continuous {mon.capitalize()} +{out}"

    # # --- ICE Strips (Quarters & Seasons) ---
    # # Only for /BPM, /PPM, /GWM
    # strip_quarters = [
    #     ('F', 'H', 'Q1'), ('J', 'M', 'Q2'), ('N', 'U', 'Q3'), ('X', 'Z', 'Q4')
    # ]
    # strip_roots = ["/BPM", "/PPM", "/GWM"]
    # for root in strip_roots:
    #     desc = AVAILABLE_COMMODITIES.get(root, root)
    #     for yys in year_suffixes:
    #         for m1, m2, q in strip_quarters:
    #             symbol = f"{root}{m1}{yys}-{m2}{yys}"
    #             variations[symbol] = f"{desc} Quarter {q} 20{yys}"

    # # --- EIA Weekly Storage, Stocks, Indices ---
    # eia_variations = {
    #     "#EIANGSTOREAST": "Weekly NG Storage East",
    #     "#EIANGSTORWEST": "Weekly NG Storage West",
    #     "#EIANGSTORPROD": "Weekly NG Storage Producing",
    #     "#EIANGSTORTL48": "Weekly NG Storage Total",
    #     "#EIKCRDTOW": "Crude Oil Stocks",
    #     "#EIKDFOTOW": "Distillate Fuel Oil Stocks",
    #     "#EIKTMOTOW": "Motor Gasoline Stocks",
    #     "#EIITPITOW": "Total Product Imports",
    #     "#EIRCOITOW": "Crude Oil Inputs",
    #     "#EIAWCEIMUS2W": "US Crude Oil Imports"
    # }
    # variations.update(eia_variations)

    # # --- Indices ---
    # stock_variations = {
    #     "$DJI": "Dow Jones Industrial Average",
    #     "$SP500": "S&P 500 Index"
    # }
    # variations.update(stock_variations)

    # # --- Options & Spreads (only as root/alias, not full chain) ---
    # options_variations = {
    #     "/GNG_SHORT": "Short Term Natural Gas Options",
    #     "/GCL_SHORT": "Short Term Crude Oil Options",
    #     "/GLOWEEK1": "Week 1 Crude Oil Options",
    #     "/GLOWEEK2": "Week 2 Crude Oil Options",
    #     "/GLOWEEK3": "Week 3 Crude Oil Options",
    #     "/GLOWEEK4": "Week 4 Crude Oil Options",
    #     "/GLOWEEK5": "Week 5 Crude Oil Options",
    #     "/GONWEEK1": "Week 1 Natural Gas Options",
    #     "/EVWEEK1": "Week 1 S&P 500 Options",
    #     "/6EWEEK2": "Week 2 Euro FX Options",
    #     "/GCL_SPREAD1": "Crude Oil 1M Spread Options",
    #     "/GCL_SPREAD6": "Crude Oil 6M Spread Options",
    #     "/GCL_SPREAD12": "Crude Oil 12M Spread Options",
    #     "/GRB_SPREAD1": "RBOB 1M Spread Options",
    #     "/GHO_SPREAD1": "Heating Oil 1M Spread Options"
    # }
    # variations.update(options_variations)

    # # --- Calendar Spreads ---
    # # Example: /GCLX25Z25, /GCL_SPREAD1, etc.
    # spread_roots = [
    #     ("/GCL", "Crude Oil"), 
    #     # ("/GNG", "Natural Gas"),
    #     # ("/GRB", "RBOB"), ("/GHO", "Heating Oil"), ("/ZC", "Corn")
    # ]
    # for root, desc in spread_roots:
    #     for m1 in MONTH_CODES:
    #         for m2 in MONTH_CODES:
    #             for y1 in year_suffixes:
    #                 for y2 in year_suffixes:
    #                     if (m1 != m2 or y1 != y2):
    #                         variations[f"{root}{m1}{y1}{m2}{y2}"] = f"{desc} Spread {m1}{y1}-{m2}{y2}"

    # --- Add all variations to the main dictionary ---
    AVAILABLE_COMMODITIES.update(variations)
    return AVAILABLE_COMMODITIES

# Categorize commodities by exchange/root symbol prefix
def categorize_commodities(commodities):
    categories = defaultdict(list)
    for symbol, desc in commodities.items():
        if symbol.startswith('/G'):  # NYMEX
            categories["NYMEX"].append((symbol, desc))
        # elif symbol.startswith(('/Z', '/EH', '/K')):
        #     categories["CBOT"].append((symbol, desc))
        # elif symbol.startswith(('/6', '/E', '/F')):
        #     categories["CME FX"].append((symbol, desc))
        # elif symbol.startswith(('/BRN', '/GAS', '/WBS')):
        #     categories["ICE Futures Europe"].append((symbol, desc))
        # elif symbol.startswith('/1s'):
        #     categories["ICE Futures US"].append((symbol, desc))
        # elif symbol.startswith(('/O', '/S')):
        #     categories["DME/TOCOM"].append((symbol, desc))
        # elif symbol.startswith('#EI'):
        #     categories["EIA Weekly"].append((symbol, desc))
        # elif symbol.startswith('$'):
        #     categories["Indices"].append((symbol, desc))
        # else:
        #     categories["Other"].append((symbol, desc))
    return categories

# Month code mapping
MONTH_CODES_PARSE = {
    'F': 'Jan', 'G': 'Feb', 'H': 'Mar', 'J': 'Apr', 'K': 'May', 'M': 'Jun',
    'N': 'Jul', 'Q': 'Aug', 'U': 'Sep', 'V': 'Oct', 'X': 'Nov', 'Z': 'Dec'
}

# Month string to month number mapping
MONTH_STR_TO_NUM = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

def parse_contract_symbol(symbol):
    """
    Extracts month and year from a futures symbol like '/GCLK25'.
    Returns (month, year) as integers, or (None, None) if parsing fails.
    """
    try:
        if not symbol or len(symbol) < 5:
            return None, None

        year_suffix = symbol[-2:]       # e.g., '25'
        month_code = symbol[-3:-2]      # e.g., 'K' from '/GCLK25'

        MONTH_CODES_PARSE = {
            'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
            'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
        }

        if not year_suffix.isdigit():
            return None, None

        month = MONTH_CODES_PARSE.get(month_code.upper())
        if not month:
            return None, None

        year = int("20" + year_suffix)  # Converts '25' -> 2025

        return month, year
    except Exception:
        return None, None

def get_month_number(month):
    """Handle both string and integer month formats."""
    if isinstance(month, int):
        return month if 1 <= month <= 12 else None
    if isinstance(month, str):
        month = month.strip().title()
        if month.isdigit():
            month = int(month)
            return month if 1 <= month <= 12 else None
        return MONTH_STR_TO_NUM.get(month)
    return None

def check_expiry(df, contract_month, contract_year, symbol):
    """
    Adds an 'expired' column to the DataFrame based on contract month/year.
    Returns 1 if expired, else 0.
    """
    try:
        contract_month_num = get_month_number(contract_month)
        contract_year = int(contract_year)
        if contract_year < 100:
            contract_year += 2000

        today = datetime.now()
        current_year = today.year
        current_month = today.month

        if contract_month_num is None:
            expired = 0
        elif (contract_year < current_year) or (contract_year == current_year and contract_month_num <= current_month):
            expired = 1
        else:
            expired = 0

    except Exception as e:
        print(f"Exception: {e}")
        expired = 0

    df['expired'] = expired
    df['Symbol'] = symbol
    return df

def get_next_commodity_symbol(commodity_symbol):
    """
    
    Args:
        commodity_data (pd.DataFrame): DataFrame containing commodity data.
        commodity_symbol (str): The symbol of the commodity.
    
    Returns:
        str: The next contract symbol, or None if not expired.
    """
        
    # Extract the letters and numbers from the symbol
    letters = ''.join([c for c in commodity_symbol if not c.isdigit()])
    numbers = ''.join([c for c in commodity_symbol if c.isdigit()])
    
    # Extract the last two digits, convert to integer, add 1, and convert back to string
    new_year = str(int(numbers[-2:]) + 1) if numbers else ""
    
    # Combine the letters and the new number to form the next symbol
    commodity_next_symbol = letters + new_year
    
    return commodity_next_symbol

def merge_old_and_new_contract(old_df, new_df):
    # Ensure Date columns are datetime
    old_df['Date'] = pd.to_datetime(old_df['Date'])
    new_df['Date'] = pd.to_datetime(new_df['Date'])

    # Sort both by Date
    old_df.sort_values("Date", inplace=True)
    new_df.sort_values("Date", inplace=True)

    # Combine old and filtered new data
    combined_df = pd.concat([old_df, new_df]).sort_values('Date')
    return combined_df
 
def append_new_after_old(old_df, new_df):
    # Ensure Date columns are datetime
    old_df['Date'] = pd.to_datetime(old_df['Date'])
    new_df['Date'] = pd.to_datetime(new_df['Date'])

    # Find the last date in old_df
    last_date_old_df = old_df['Date'].max()

    # Filter new_df to only include rows after the last date of old_df
    new_df_filtered = new_df[new_df['Date'] > last_date_old_df]

    # Append the filtered new_df to old_df
    combined_df = pd.concat([old_df, new_df_filtered]).sort_values('Date')
    
    return combined_df

def get_expiry_date(symbol):
    month, year = parse_contract_symbol(symbol)
    if month is None or year is None:
        return None  # or a default datetime or np.nan
    # Find the month before the contract month
    if month == 1:
        expiry_month = 12
        expiry_year = year - 1
    else:
        expiry_month = month - 1
        expiry_year = year

    # Get US calendar
    nymex = mcal.get_calendar('Financial_Markets_US')
    # 25th of expiry month
    twenty_fifth = datetime(expiry_year, expiry_month, 25)
    # Find valid business days in that month
    sched = nymex.schedule(start_date=f'{expiry_year}-{expiry_month:02d}-01',
                           end_date=f'{expiry_year}-{expiry_month:02d}-28')
    business_days = sched.index

    # Find the last business day <= 25th
    last_bday = max([d for d in business_days if d <= twenty_fifth])
    # Go back three business days
    idx = business_days.get_loc(last_bday)
    expiry_idx = idx - 3
    expiry_date = business_days[expiry_idx].date()
    return expiry_date





 
