import pandas as pd


def get_dataset():
    brazil_df = pd.read_csv("data/brazil_matches.csv")
    brazil_df = brazil_df.drop(
        columns=["knockout", "stage", "tournament_name", "tournament_year"]
    )
    brazil_df["fold"] = "brazil"
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
    return pd.concat(
        [brazil_df, europe_df, international_df], axis=0, ignore_index=True
    )
