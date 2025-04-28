from datetime import datetime
from collections import defaultdict

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


def get_dataset(with_features=False):
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
    if with_features:
        df = generate_features(df[
            [
                "date"
                "team_id",
                "opponent_id",
                "team_at_home",
                "opponent_at_home",
                "team_score",
                "opponent_score",
                "fold",
            ]
        ])
        df = df.drop(columns=["date"])
        return df
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


def _count_consecutive_strings_reverse(results: list[str], strings: list[str]) -> int:
    count = 0
    for el in results[::-1]:
        if el in strings:
            count += 1
        else:
            return count
    return count


def generate_features(df):
    last_match = defaultdict(str)
    scores = defaultdict(list)
    scores_against = defaultdict(list)
    results = defaultdict(list)
    results_against = defaultdict(lambda: defaultdict(list))
    for index, row in df.iterrows():
        team_id = row["team_id"]
        opponent_id = row["opponent_id"]
        # Get features
        df.at[index, "team_days_since_last_match"] = (
            (datetime.strptime(row["date"], "%Y-%m-%d") - datetime.strptime(last_match[team_id], "%Y-%m-%d")).days
            if last_match[team_id]
            else 0.0
        )
        df.at[index, "opponent_days_since_last_match"] = (
            (datetime.strptime(row["date"], "%Y-%m-%d") - datetime.strptime(last_match[opponent_id], "%Y-%m-%d")).days
            if last_match[opponent_id]
            else 0.0
        )
        df.at[index, "team_5m_score_avg"] = sum(scores[team_id][-5:]) / len(scores[team_id][-5:]) if len(scores[team_id][-5:]) else 0.0
        df.at[index, "opponent_5m_score_avg"] = sum(scores[opponent_id][-5:]) / len(scores[opponent_id][-5:]) if len(scores[opponent_id][-5:]) else 0.0
        df.at[index, "team_5m_score_against_opponent_avg"] = sum(scores_against[team_id][-5:]) / len(scores_against[team_id][-5:]) if len(scores_against[team_id][-5:]) else 0.0
        df.at[index, "opponent_5m_score_against_team_avg"] = sum(scores_against[opponent_id][-5:]) / len(scores_against[opponent_id][-5:]) if len(scores_against[opponent_id][-5:]) else 0.0
        df.at[index, "team_consecutive_wins_overall"] = _count_consecutive_strings_reverse(results[team_id], ["win"])
        df.at[index, "opponent_consecutive_wins_overall"] = _count_consecutive_strings_reverse(results[opponent_id], ["win"])
        df.at[index, "team_consecutive_non_losses_overall"] = _count_consecutive_strings_reverse(results[team_id], ["win", "draw"])
        df.at[index, "opponent_consecutive_non_losses_overall"] = _count_consecutive_strings_reverse(results[opponent_id], ["win", "draw"])
        df.at[index, "team_10m_count_wins_overall"] = len([result for result in results[team_id][-10:] if result == "win"])
        df.at[index, "team_10m_count_draws_overall"] = len([result for result in results[team_id][-10:] if result == "draw"])
        df.at[index, "team_10m_count_losses_overall"] = len([result for result in results[team_id][-10:] if result == "loss"])
        df.at[index, "opponent_10m_count_wins_overall"] = len([result for result in results[opponent_id][-10:] if result == "win"])
        df.at[index, "opponent_10m_count_draws_overall"] = len([result for result in results[opponent_id][-10:] if result == "draw"])
        df.at[index, "opponent_10m_count_losses_overall"] = len([result for result in results[opponent_id][-10:] if result == "loss"])
        df.at[index, "team_consecutive_wins_against_opponent"] = _count_consecutive_strings_reverse(results_against[team_id][opponent_id], ["win"])
        df.at[index, "opponent_consecutive_wins_against_opponent"] = _count_consecutive_strings_reverse(results_against[opponent_id][team_id], ["win"])
        df.at[index, "team_consecutive_non_losses_overall"] = _count_consecutive_strings_reverse(results_against[team_id][opponent_id], ["win", "draw"])
        df.at[index, "opponent_consecutive_non_losses_against_opponent"] = _count_consecutive_strings_reverse(results_against[opponent_id][team_id], ["win", "draw"])
        df.at[index, "team_10m_count_wins_against_opponent"] = len([result for result in results_against[team_id][opponent_id][-10:] if result == "win"])
        df.at[index, "team_10m_count_draws_against_opponent"] = len([result for result in results_against[team_id][opponent_id][-10:] if result == "draw"])
        df.at[index, "team_10m_count_losses_against_opponent"] = len([result for result in results_against[team_id][opponent_id][-10:] if result == "loss"])
        df.at[index, "opponent_10m_count_wins_against_opponent"] = len([result for result in results_against[opponent_id][team_id][-10:] if result == "win"])
        df.at[index, "opponent_10m_count_draws_against_opponent"] = len([result for result in results_against[opponent_id][team_id][-10:] if result == "draw"])
        df.at[index, "opponent_10m_count_losses_against_opponent"] = len([result for result in results_against[opponent_id][team_id][-10:] if result == "loss"])
        # Append information
        last_match[team_id] = row["date"]
        last_match[opponent_id] = row["date"]
        scores[team_id].append(int(row["team_score"]))
        scores[opponent_id].append(int(row["opponent_score"]))
        scores_against[team_id].append(int(row["opponent_score"]))
        scores_against[opponent_id].append(int(row["team_score"]))
        if row["team_score"] > row["opponent_score"]:
            results[team_id].append("win")
            results[opponent_id].append("loss")
            results_against[team_id][opponent_id].append("win")
            results_against[opponent_id][team_id].append("loss")
        elif row["team_score"] == row["opponent_score"]:
            results[team_id].append("draw")
            results[opponent_id].append("draw")
            results_against[team_id][opponent_id].append("draw")
            results_against[opponent_id][team_id].append("draw")
        else:
            results[team_id].append("loss")
            results[opponent_id].append("win")
            results_against[team_id][opponent_id].append("loss")
            results_against[opponent_id][team_id].append("win")
    return df