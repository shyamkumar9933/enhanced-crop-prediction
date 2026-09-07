import pandas as pd


BASE_FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

MODEL_FEATURES = BASE_FEATURES + ["season"]


def add_season(df):
    """
    Add a simple season feature derived from temperature.
    The same transformation is used during training and prediction.
    """
    data = df.copy()

    def get_season(temp):
        if temp < 20:
            return "Winter"
        elif temp < 28:
            return "Spring"
        elif temp < 34:
            return "Summer"
        else:
            return "Monsoon"

    data["season"] = data["temperature"].apply(get_season)

    return data


def clean_data(df):
    """
    Clean numeric input columns by converting them to numeric
    values and filling missing values with column medians.
    """
    data = df.copy()

    for column in BASE_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].fillna(data[column].median())

    return data


def prepare_features(df):
    """
    Prepare the final feature matrix used by the ML model.
    """
    data = clean_data(df)
    data = add_season(data)

    # Convert season to a stable numeric representation.
    season_mapping = {
        "Winter": 0,
        "Spring": 1,
        "Summer": 2,
        "Monsoon": 3,
    }

    data["season"] = data["season"].map(season_mapping)

    return data[MODEL_FEATURES]


def prepare_single_input(values):
    """
    Convert one prediction input dictionary into a DataFrame.
    """
    row = pd.DataFrame([values])

    return prepare_features(row)
