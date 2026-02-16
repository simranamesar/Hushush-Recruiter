import pandas as pd
from models import (
    get_randomforest_github_table,
    get_gradientboosting_codeforces_table,
    get_decisiontree_github_table,
    get_randomforest_codeforces_table
)

def get_final_shortlist():
    """Combine all model tables for HR final shortlist"""
    # Load tables
    rf_table = get_randomforest_github_table()
    gb_table = get_gradientboosting_codeforces_table()
    dt_table = get_decisiontree_github_table()
    cf_table = get_randomforest_codeforces_table()

    # Combine
    combined = pd.concat(
        [rf_table, gb_table, dt_table, cf_table],
        ignore_index=True
    )

    combined["Candidate"] = (
        combined["user_name"]
        .fillna(combined["username"])
        .fillna(combined["github_username"])
    )

    combined["Stage"] = "Round 2 – Technical Interview"

    return combined[["Candidate", "Stage"]].drop_duplicates()

def check_candidate_status(gh_username):
    """Check if candidate exists in any model table"""
    rf_table = get_randomforest_github_table()
    gb_table = get_gradientboosting_codeforces_table()
    dt_table = get_decisiontree_github_table()
    cf_table = get_randomforest_codeforces_table()

    all_tables = [rf_table, gb_table, dt_table, cf_table]
    combined_df = pd.concat(all_tables, ignore_index=True)

    combined_df["name"] = combined_df[["user_name", "username", "github_username"]].bfill(axis=1).iloc[:, 0]
    combined_df["name"] = combined_df["name"].astype(str)

    candidate_row = combined_df[combined_df["name"].str.lower() == gh_username.lower()]

    if candidate_row.empty:
        return None
    
    return candidate_row.iloc[0]
