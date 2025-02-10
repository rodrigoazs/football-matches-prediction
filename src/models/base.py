from abc import ABC, abstractmethod

import pandas as pd


class BaseMatchPredictor(ABC):
    def _reverse_matches(self, X, y):
        """Reverse the matches in the dataset."""
        X_rows, y_rows = [], []
        for data, new_rows in zip([X, y], [X_rows, y_rows]):
            for index, row in data.iterrows():
                new_rows.append(row)
                new_row = row.copy()
                for col in data.columns:
                    if col.startswith("team_"):
                        opponent_col = col.replace("team_", "opponent_")
                        new_row[col], new_row[opponent_col] = (
                            row[opponent_col],
                            row[col],
                        )
                new_rows.append(new_row)
        return pd.DataFrame(X_rows).reset_index(drop=True), pd.DataFrame(
            y_rows
        ).reset_index(drop=True)
