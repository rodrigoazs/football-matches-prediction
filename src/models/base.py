from abc import ABC, abstractmethod

import pandas as pd


class BaseMatchPredictor(ABC):
    def _sort_columns(self, cols, consider_id=True):
        if consider_id and ("team_id" not in cols or "opponent_id" not in cols):
            raise ValueError("Columns must contain 'team_id' and 'opponent_id'.")
        cols = sorted(cols)
        team_cols, opponent_cols, common_cols = [], [], []
        for col in cols:
            if col == "team_id" or col == "opponent_id":
                continue
            elif col.startswith("team_"):
                team_cols.append(col[5:])
            elif col.startswith("opponent_"):
                opponent_cols.append(col[9:])
            else:
                common_cols.append(col)
        if team_cols != opponent_cols:
            raise ValueError(
                "Columns must have matching 'team_' and 'opponent_' columns."
            )
        sorted_cols = []
        for team_col, opponent_col in zip(team_cols, opponent_cols):
            sorted_cols.extend([f"team_{team_col}", f"opponent_{opponent_col}"])
        id_cols = ["team_id", "opponent_id"] if consider_id else []
        return id_cols + sorted_cols + common_cols

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
        X = pd.DataFrame(X_rows).reset_index(drop=True)
        y = pd.DataFrame(y_rows).reset_index(drop=True)
        X = X[self._sort_columns(X.columns, consider_id=True)]
        y = y[self._sort_columns(y.columns, consider_id=False)]
        return X, y
