import streamlit as st
import pandas as pd

# Define tag_logic and tag_allocation_rules globally
tag_logic = {
    1:  [1, 0, 0, 0, 0],
    2:  [1, 1, 0, 0, 0],
    3:  [1, 1, 1, 0, 0],
    4:  [1, 1, 1, 1, 0],
    5:  [1, 1, 1, 1, 1],
    6:  [2, 1, 1, 1, 1],
    7:  [2, 2, 1, 1, 1],
    8:  [2, 2, 2, 1, 1],
    9:  [2, 2, 2, 2, 1],
    10: [2, 2, 2, 2, 2],
    11: [3, 2, 2, 2, 2],
    12: [3, 3, 2, 2, 2],
    13: [3, 3, 3, 2, 2],
    14: [3, 3, 3, 3, 2],
    15: [3, 3, 3, 3, 3],
    16: [4, 3, 3, 3, 3],
    17: [4, 4, 3, 3, 3],
    18: [4, 4, 4, 3, 3],
    19: [4, 4, 4, 4, 3],
    20: [4, 4, 4, 4, 4],
    21: [5, 4, 4, 4, 4],
    22: [5, 5, 4, 4, 4],
    23: [5, 5, 5, 4, 4],
    24: [5, 5, 5, 5, 4],
    25: [5, 5, 5, 5, 5]
}

# Convert tag_logic to tag_allocation_rules format for both D and B series
tag_allocation_rules = {'D': {}, 'B': {}}
for tag, values in tag_logic.items():
    tag_allocation_rules['D'][tag] = {str(i+1): value for i, value in enumerate(values)}
    tag_allocation_rules['B'][tag] = {str(i+1): value for i, value in enumerate(values)}

def process_csv(file):
    """Process the strategy CSV file to count the number of times each user appears in each StrategyTag."""
    try:
        df = pd.read_csv(file, skiprows=5)
        df.columns = df.columns.str.strip()
        required_cols = ["StrategyTag", "User Account"]
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Error: '{col}' column not found in dataset.")
                return None
        df = df[required_cols]
        df = df[~df["StrategyTag"].isin(["DEFAULT", "ZZZ"])]
        valid_strategies = ['B1', 'B2', 'B3', 'B4', 'B5', 'D1', 'D2', 'D3', 'D4', 'D5']
        df = df[df["StrategyTag"].isin(valid_strategies)]
        user_accounts_split = df["User Account"].str.split(";", expand=True)
        df_melted = user_accounts_split.melt(ignore_index=False, var_name="User Column", value_name="User Account")
        df_melted.dropna(subset=["User Account"], inplace=True)
        df_melted.reset_index(inplace=True)
        df_melted = df_melted.merge(df[["StrategyTag"]], left_on="index", right_index=True, how="left")
        df_melted[["User Type", "User Value"]] = df_melted["User Account"].str.split("=", expand=True)
        df_melted["User Value"] = pd.to_numeric(df_melted["User Value"], errors="coerce").fillna(1)
        df_counts = df_melted.groupby(["StrategyTag", "User Type"])["User Value"].sum().reset_index()
        df_pivot = df_counts.pivot(index="StrategyTag", columns="User Type", values="User Value").fillna(0)
        df_pivot = df_pivot.reset_index()
        return df_pivot
    except pd.errors.EmptyDataError:
        st.error("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        st.error("Error: Unable to parse the file. Ensure it's a valid CSV.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def get_allocation_tag(allocation, series):
    """Map an allocation to a Tag based on the provided table and series (B or D)."""
    if series == 'D':
        capital_deployed_tags = [
            (4000000, 1), (8000000, 2), (12000000, 3), (16000000, 4), (20000000, 5),
            (24000000, 6), (28000000, 7), (32000000, 8), (36000000, 9), (40000000, 10)
        ]
        for capital_deployed, tag in capital_deployed_tags:
            if allocation == capital_deployed:
                return tag, None
        return None, f"No exact match found for allocation {allocation} in Capital Deployed values."
    elif series == 'B':
        capital_deployed_tags = [
            (4000000, 2), (6000000, 3), (8000000, 4), (10000000, 5), (12000000, 6),
            (14000000, 7), (16000000, 8), (18000000, 9), (20000000, 10), (22000000, 11),
            (24000000, 12), (26000000, 13), (28000000, 14), (30000000, 15), (32000000, 16),
            (34000000, 17), (36000000, 18), (38000000, 19), (40000000, 20), (42000000, 21),
            (44000000, 22), (46000000, 23), (48000000, 24), (50000000, 25)
        ]
        for capital_deployed, tag in capital_deployed_tags:
            if allocation == capital_deployed:
                return tag, None
        return None, f"No exact match found for allocation {allocation} in Capital Deployed values."
    else:
        return None, "Invalid series"

def allocate_strategies(file, user_types, strategies, series):
    """Process the user settings CSV file to compute allocations and return a DataFrame and a dictionary of {userid: allocation, tag}."""
    try:
        df = pd.read_csv(file, skiprows=6)
    except pd.errors.EmptyDataError:
        st.error("Error: The user settings file is empty.")
        return None, None
    except pd.errors.ParserError:
        st.error("Error: Unable to parse the user settings file Ensure it's a valid CSV.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while reading user settings file: {e}")
        return None, None

    df.columns = df.columns.str.strip()
    required_cols = ['User ID', 'Telegram ID(s)']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Error: The following required columns are missing in the user settings file: {missing_cols}")
        return None, None

    df['User ID'] = df['User ID'].astype(str)
    user_types = [str(user_type) for user_type in user_types]
    df['Telegram ID(s)'] = pd.to_numeric(df['Telegram ID(s)'], errors='coerce').fillna(0).astype(int)
    df['ALLOCATION'] = df['Telegram ID(s)'] * 100
    allocation_dict = dict(zip(df['User ID'], df['ALLOCATION']))
    user_tags = {}
    for user_id, allocation in allocation_dict.items():
        tag, error = get_allocation_tag(allocation, series)
        if tag is not None:
            user_tags[user_id] = tag
        else:
            st.warning(f"User {user_id}: {error}")

    df_filtered = df[['User ID', 'ALLOCATION']].copy()
    df_filtered.rename(columns={'User ID': 'userId'}, inplace=True)
    all_users_df = pd.DataFrame({'userId': user_types})
    df_filtered = all_users_df.merge(df_filtered, on="userId", how="left").fillna({'ALLOCATION': 0})

    allocation_df = pd.DataFrame({'userId': user_types})
    allocation_df['ALLOCATION'] = df_filtered['ALLOCATION']
    for strategy in strategies:
        allocation_df[strategy] = 0

    for user_type in user_types:
        if user_type in user_tags and user_tags[user_type] in tag_allocation_rules[series]:
            rule = tag_allocation_rules[series][user_tags[user_type]]
            for idx, strategy in enumerate(strategies, 1):
                allocation_df.loc[allocation_df['userId'] == user_type, strategy] = rule[str(idx)]

    return allocation_df, (allocation_dict, user_tags)

def main():
    st.title("Strategy Allocation Checking")
    
    stoxo_file = st.file_uploader("Upload Strategy CSV file ", type=["csv"])
    allocation_file = st.file_uploader("Upload User Settings CSV file ", type=["csv"])

    if stoxo_file is not None and allocation_file is not None:
        stoxo_df = process_csv(stoxo_file)
        if stoxo_df is None:
            return

        valid_strategies = ['B1', 'B2', 'B3', 'B4', 'B5', 'D1', 'D2', 'D3', 'D4', 'D5']
        available_strategies = stoxo_df["StrategyTag"].unique()
        strategies = [s for s in available_strategies if s in valid_strategies]

        if any(s.startswith('B') for s in strategies):
            series = 'B'
            strategies = ['B1', 'B2', 'B3', 'B4', 'B5']
        elif any(s.startswith('D') for s in strategies):
            series = 'D'
            strategies = ['D1', 'D2', 'D3', 'D4', 'D5']
        else:
            st.error("Error: No valid strategies (B1-B5 or D1-D5) found in the data.")
            return

        stoxo_df = stoxo_df[stoxo_df["StrategyTag"].isin(strategies)]
        user_types = [col for col in stoxo_df.columns if col != "StrategyTag"]

        allocation_df, (allocation_dict, user_tags) = allocate_strategies(allocation_file, user_types, strategies, series)
        if allocation_df is None:
            return

        st.subheader("User ID to Allocation Dictionary")
        st.write(allocation_dict)

        output_data = []
        for user_type in user_types:
            row_data = {
                "userId": user_type,
                "ALLOCATION": allocation_df[allocation_df["userId"] == user_type]["ALLOCATION"].iloc[0] if not allocation_df[allocation_df["userId"] == user_type].empty else 0
            }
            alloc_row = allocation_df[allocation_df["userId"] == str(user_type)]
            alloc_values = {strategy: alloc_row[strategy].iloc[0] for strategy in strategies} if not alloc_row.empty else {strategy: 0 for strategy in strategies}
            
            user_tag = user_tags.get(user_type, None)
            expected_strategies = set()
            if user_tag and series in tag_allocation_rules and user_tag in tag_allocation_rules[series]:
                rule = tag_allocation_rules[series][user_tag]
                for idx, strategy in enumerate(strategies, 1):
                    if rule[str(idx)] > 0:
                        expected_strategies.add(strategy)
            
            for strategy in strategies:
                stoxo_row = stoxo_df[stoxo_df["StrategyTag"] == strategy]
                stoxo_calc = stoxo_row[user_type].iloc[0] if not stoxo_row.empty and user_type in stoxo_row.columns else 0
                calculated_alloc = alloc_values[strategy]
                values_equal = (calculated_alloc == stoxo_calc)
                row_data[f"{strategy}_Match"] = values_equal
                row_data[f"{strategy}_Calculated allocation"] = calculated_alloc
                row_data[f"{strategy}_Stoxo calculation"] = stoxo_calc
            
            output_data.append(row_data)

        columns = ["userId", "ALLOCATION"]
        columns.extend([f"{strategy}_Match" for strategy in strategies])
        for strategy in strategies:
            columns.extend([f"{strategy}_Calculated allocation", f"{strategy}_Stoxo calculation"])
        output_df = pd.DataFrame(output_data, columns=columns)

        st.subheader("Processed Output")
        st.dataframe(output_df)

        # Provide download button for the output Excel file
        output_file = "processed_output.xlsx"
        output_df.to_excel(output_file, index=False)
        with open(output_file, "rb") as file:
            st.download_button(
                label="Download Processed Output",
                data=file,
                file_name="processed_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
