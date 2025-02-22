import json

import pandas as pd

from online_inference_pipeline.utils import load_object_from_s3


class Transformer:
    def __init__(self):
        self.DROP_COLUMNS = []
        self.BINARY_FEATURES = [
            "housing",
            "loan",
            "default",
        ]
        self.ONE_HOT_ENCODE_COLUMNS = [
            "marital",
            "job",
            "education",
            "poutcome",
            "contact",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(self.DROP_COLUMNS, axis=1)
        df = self._map_binary_column_to_int(df)
        df = self._map_month_to_int(df)
        df = self._one_hot_encoding(df)

        return df

    def _map_binary_column_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.BINARY_FEATURES:
            df[col] = df[col].map({"yes": 1, "no": 0})
        return df

    def _map_month_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        month_mapping = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        df["month"] = df["month"].map(month_mapping)

        return df

    def _one_hot_encoding(self, df):
        encoder = load_object_from_s3(
            bucket_name="dataset-mlops-robert", model_key="one_hot_encoder.joblib"
        )
        encoded_df = encoder.transform(df[self.ONE_HOT_ENCODE_COLUMNS])
        df = df.drop(columns=self.ONE_HOT_ENCODE_COLUMNS)
        df = pd.concat([df, encoded_df], axis=1)

        return df


def preprocess(event) -> pd.DataFrame:
    if "body" in event:
        body = json.loads(event["body"])  # Convert string to dictionary
    else:
        body = event  # If it's direct invocation, use as is

    return pd.DataFrame([body])
