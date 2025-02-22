import pandas as pd

from online_inference_pipeline.transform import Transformer


def test_map_binary_column_to_int():
    transformer = Transformer()
    df = pd.DataFrame(
        {
            "housing": ["yes", "no"],
            "loan": ["no", "yes"],
            "default": ["no", "yes"],
        }
    )
    transformed_df = transformer._map_binary_column_to_int(df)

    assert transformed_df["housing"].tolist() == [1, 0]
    assert transformed_df["loan"].tolist() == [0, 1]
    assert transformed_df["default"].tolist() == [0, 1]


def test_map_month_to_int():
    transformer = Transformer()
    df = pd.DataFrame({"month": ["jan", "feb", "mar"]})
    transformed_df = transformer._map_month_to_int(df)

    assert transformed_df["month"].tolist() == [1, 2, 3]
