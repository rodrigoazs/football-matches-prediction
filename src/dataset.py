from datetime import datetime

import pandas as pd


def determine_date(row):
    date_row = row["date"]
    year_row = row["year"]
    try:
        return datetime.strptime(f"{date_row} {year_row}", "%A, %B %d %Y").strftime(
            "%Y-%m-%d"
        )
    except:
        return datetime.strptime(f"{date_row}", "%m/%d/%Y").strftime("%Y-%m-%d")


def get_dataset():
    brazil_df = pd.read_csv("data/brazil_matches.csv")
    brazil_df = brazil_df.drop(
        columns=["knockout", "stage", "tournament_name", "tournament_year"]
    )
    brazil_df["fold"] = "brazil"
    libertadores_df = pd.read_csv("data/libertadores_matches.csv")
    libertadores_df = libertadores_df.drop(
        columns=["knockout", "stage", "tournament_name", "tournament_year"]
    )
    libertadores_df["fold"] = "libertadores"
    europe_df = pd.read_csv("data/europe_matches.csv")
    europe_df = europe_df.rename(
        columns={
            "home_club_name": "home_team",
            "away_club_name": "away_team",
            "home_club_goals": "home_score",
            "away_club_goals": "away_score",
        }
    )
    europe_df["neutral"] = False
    europe_df = europe_df[
        ["date", "home_team", "home_score", "away_score", "away_team", "neutral"]
    ]
    europe_df["fold"] = "europe"
    international_df = pd.read_csv("data/national_teams.csv")
    international_df = international_df.drop(
        columns=["tournament_name", "tournament_year"]
    )
    international_df["fold"] = "international"
    mls_df = pd.read_csv("data/mls_matches.csv")
    mls_df = mls_df.rename(
        columns={
            "home": "home_team",
            "away": "away_team",
        }
    )
    mls_df = mls_df[
        ["date", "year", "home_team", "home_score", "away_score", "away_team"]
    ]
    mls_df["date"] = mls_df.apply(determine_date, axis=1)
    mls_df = mls_df.drop(columns=["year"])
    mls_df["neutral"] = False
    mls_df["fold"] = "mls"
    df = pd.concat(
        [brazil_df, libertadores_df, mls_df, europe_df, international_df],
        axis=0,
        ignore_index=True,
    )
    df["team_at_home"] = df["neutral"].apply(lambda x: 0.0 if x else 1.0)
    df["opponent_at_home"] = 0.0
    df = df.rename(
        columns={
            "home_team": "team_id",
            "away_team": "opponent_id",
            "home_score": "team_score",
            "away_score": "opponent_score",
        }
    )
    df = df.sort_values(by="date", ascending=True)
    return df[
        [
            "team_id",
            "opponent_id",
            "team_at_home",
            "opponent_at_home",
            "team_score",
            "opponent_score",
            "fold",
        ]
    ].reset_index(drop=True)
