import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from interfaces.classification_evaluator_interface import ClassificationEvaluatorInterface


class SklearnClassificationEvaluator(ClassificationEvaluatorInterface):
    """Random Forest classifier with 80/20 train/test split."""

    def __init__(self, random_state: int = 42) -> None:
        self._random_state = random_state

    def evaluate(self, df: pd.DataFrame, target_column: str) -> float:
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not in DataFrame. "
                f"Available: {df.columns.tolist()}"
            )

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self._random_state
        )

        model = RandomForestClassifier(random_state=self._random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return float(accuracy_score(y_test, y_pred))