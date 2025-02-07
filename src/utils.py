import pandas as pd


def swap_dataset(df):
    swaped_df = pd.DataFrame()
    swaped_df["date"] = df["date"]
    swaped_df["home_team"] = df["away_team"]
    swaped_df["home_score"] = df["away_score"]
    swaped_df["away_score"] = df["home_score"]
    swaped_df["away_team"] = df["home_team"]
    swaped_df["home_rating"] = df["away_rating"]
    swaped_df["away_rating"] = df["home_rating"]
    return swaped_df
