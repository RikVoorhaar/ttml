import os.path
import pandas as pd
import numpy as np
import scipy.io.arff

DATASET_FOLDER = "data"


def cat_to_num(df, excluded=None):
    """Casts columns with 'object' dtype to numerical, ignoring any column in 'excluded'"""
    if excluded is None:
        excluded = []
    categorical_features = []
    for i, (column, dtype) in enumerate(df.dtypes.iteritems()):
        if dtype == "object" and column not in excluded:
            categorical_features.append(i)
            df[column] = df[column].astype("category").cat.codes
    return categorical_features


def prepare_ai4i2020(data_folder=DATASET_FOLDER):
    """AI4I 2020 Predictive Maintenance Dataset"""
    data_path = os.path.join(data_folder, "ai4i2020.csv")
    df = pd.read_csv(data_path)
    y = df["Machine failure"]
    X = df.iloc[:, 2:8]
    X["Type"] = X["Type"].astype("category").cat.codes
    return {
        "X": X.to_numpy(),
        "y": y.to_numpy(),
        "regression": False,
        "categorical_features": [0],
    }


def prepare_airfoil(data_folder=DATASET_FOLDER):
    """Airfoil Self-Noise Data Set"""
    data_path = os.path.join(data_folder, "airfoil_self_noise.dat")
    df = pd.read_csv(data_path, sep="\t", header=None)
    y = df[5]
    X = df.drop(5, axis=1)
    return {
        "X": X.to_numpy(),
        "y": y.to_numpy(),
        "regression": True,
        "categorical_features": [2, 3],
    }


def prepare_bank_marketing(data_folder=DATASET_FOLDER):
    """Bank Marketing Data Set"""
    data_path = os.path.join(data_folder, "bank_marketing.csv")
    df = pd.read_csv(data_path, sep=";")
    categorical_features = cat_to_num(df)
    categorical_features = categorical_features[
        :-1
    ]  # remove y as categorical feature
    X = df.drop("y", axis=1)
    y = df["y"]
    return {
        "X": X.to_numpy(),
        "y": y.to_numpy(),
        "regression": False,
        "categorical_features": categorical_features,
    }


def prepare_census_income(data_folder=DATASET_FOLDER):
    """Adult Data Set"""
    data_path = os.path.join(data_folder, "census_income.data")
    df = pd.read_csv(data_path, header=None)
    y = df[14].astype("category").cat.codes
    df.drop(14, inplace=True, axis=1)
    categorical_features = cat_to_num(df)
    return {
        "X": df.to_numpy(),
        "y": y.to_numpy(),
        "regression": False,
        "categorical_features": categorical_features,
    }


def prepare_power_plant(data_folder=DATASET_FOLDER):
    """Combined Cycle Power Plant Data Set"""
    data_path = os.path.join(data_folder, "combinedCyclePowerPlant.xlsx")
    df = pd.read_excel(data_path)
    y = df.PE
    X = df.drop("PE", axis=1)
    return {
        "X": X.to_numpy(),
        "y": y.to_numpy(),
        "regression": True,
        "categorical_features": [],
    }


def prepare_concrete(data_folder=DATASET_FOLDER):
    """Concrete Compressive Strength Data Set"""
    data_path = os.path.join(data_folder, "Concrete_Data.xls")
    df = pd.read_excel(data_path)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    return {
        "X": X.to_numpy(),
        "y": y.to_numpy(),
        "regression": True,
        "categorical_features": [],
    }


def prepare_default_credit_card(data_folder=DATASET_FOLDER):
    """default of credit card clients Data Set"""
    data_path = os.path.join(data_folder, "default_of_credit_card_clients.xls")
    df = pd.read_excel(data_path, header=1)
    df.drop("ID", axis=1, inplace=True)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    return {
        "X": X.to_numpy(),
        "y": y.to_numpy(),
        "regression": False,
        "categorical_features": [1, 2, 3],
    }


def prepare_diabetic_retinopathy(data_folder=DATASET_FOLDER):
    """Diabetic Retinopathy Debrecen Data Set Data Set"""
    data_path = os.path.join(data_folder, "diabetic_retinopathy_debrecen.arff")
    data, _ = scipy.io.arff.loadarff(data_path)
    df = pd.DataFrame(data)
    y = df["Class"].astype("category").cat.codes
    X = df.drop("Class", axis=1)
    return {
        "X": X.to_numpy(),
        "y": y.to_numpy(),
        "regression": False,
        "categorical_features": [0, 1, 18],
    }


def prepare_electrical_grid(data_folder=DATASET_FOLDER):
    """Electrical Grid Stability Simulated Data Data Set"""
    data_path = os.path.join(data_folder, "electric_grid_stability.csv")
    df = pd.read_csv(data_path)
    y = df.iloc[:, -1].astype("category").cat.codes
    X = df.iloc[:, :-1]
    return {
        "X": X.to_numpy(),
        "y": y.to_numpy(),
        "regression": False,
        "categorical_features": [],
    }


def prepare_gas_turbine(data_folder=DATASET_FOLDER):
    """Gas Turbine CO and NOx Emission Data Set Data Set"""
    from zipfile import ZipFile

    data_path = os.path.join(data_folder, "gas_turbine_emission.zip")
    datasets = []
    with ZipFile(data_path) as data_zip:
        for data in data_zip.namelist():
            df = pd.read_csv(data_zip.open(data))
            datasets.append(df)
    df = pd.concat(datasets)

    y = df["TEY"]
    X = df.drop("TEY", axis=1)
    return {
        "X": X.to_numpy(),
        "y": y.to_numpy(),
        "regression": True,
        "categorical_features": [],
    }

# WE DO NOT USE THIS DATASET ANYMORE
# def prepare_vehicle_coupon(data_folder=DATASET_FOLDER):
#     """in-vehicle coupon recommendation Data Set"""
#     data_path = os.path.join(
#         data_folder, "in-vehicle-coupon-recommendation.csv"
#     )
#     url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00603/in-vehicle-coupon-recommendation.csv"
#     df = pd.read_csv(data_path)
#     categorical_features = cat_to_num(df)
#     y = df.Y
#     X = df.drop("Y", axis=1)
#     return {
#         "X": X.to_numpy(),
#         "y": y.to_numpy(),
#         "regression": False,
#         "categorical_features": categorical_features,
#     }


def prepare_mini_boone(data_folder=DATASET_FOLDER):
    """MiniBooNE particle identification Data Set"""
    data_path = os.path.join(data_folder, "MiniBooNE_PID.txt")
    with open(data_path) as f:
        num_classes = f.readline()
        n_events, n_backgrounds = num_classes.split()
        n_events = int(n_events)
        n_backgrounds = int(n_backgrounds)

        lines_processed = []
        y = []
        for i, line in enumerate(f):
            line_processed = [float(x) for x in line.split()]
            if i < n_events:
                y.append(1)
            else:
                y.append(0)
            lines_processed.append(line_processed)

    X = np.array(lines_processed)
    y = np.array(y)
    return {
        "X": X,
        "y": y,
        "regression": False,
        "categorical_features": [],
    }


def prepare_online_shoppers(data_folder=DATASET_FOLDER):
    """Online Shoppers Purchasing Intention Dataset Data Set"""
    data_path = os.path.join(data_folder, "online_shoppers_intention.csv")
    df = pd.read_csv(data_path, delimiter=",")
    categorical_features = cat_to_num(df)
    categorical_features.extend([11, 12, 9, 16])
    categorical_features.sort()
    y = df["Revenue"]
    X = df.drop("Revenue", axis=1)
    return {
        "X": X.to_numpy(dtype=float),
        "y": y.to_numpy(),
        "regression": False,
        "categorical_features": categorical_features,
    }


# WE DO NOT USE THIS DATASET ANYMORE
# def prepare_bankruptcy(data_folder=DATASET_FOLDER):
#     """Polish companies bankruptcy data Data Set"""
#     #TODO: change this script; we want to directly open the data.zip
#     data_folder = os.path.join(data_folder, "polish_bankruptcy")
#     url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip"
#     dfs = []
#     for f in os.listdir(data_folder):
#         data_path = os.path.join(data_folder, f)
#         data, _ = scipy.io.arff.loadarff(data_path)
#         df = pd.DataFrame(data)
#         dfs.append(df)
#     df = pd.concat(dfs)
#     y = df["class"].astype("category").cat.codes
#     X = df.drop("class", axis=1)
#     return {
#         "X": X.to_numpy(),
#         "y": y.to_numpy(),
#         "regression": False,
#         "categorical_features": [],
#     }


def prepare_seismic_bumps(data_folder=DATASET_FOLDER):
    """seismic-bumps Data Set"""
    data_path = os.path.join(data_folder, "seismic-bumps.arff")
    data, _ = scipy.io.arff.loadarff(data_path)
    df = pd.DataFrame(data)
    y = df["class"].to_numpy(dtype=float)
    df.drop(["nbumps6", "nbumps7", "nbumps89", "class"], axis=1, inplace=True)

    categorical_features = cat_to_num(df)
    categorical_features.extend([11, 12])
    categorical_features.sort()
    X = df.to_numpy(dtype=float)
    return {
        "X": X,
        "y": y,
        "regression": False,
        "categorical_features": categorical_features,
    }


def prepare_shill_bidding(data_folder=DATASET_FOLDER):
    """Shill Bidding Dataset Data Set"""
    data_path = os.path.join(data_folder, "Shill_Bidding_Dataset.csv")
    df = pd.read_csv(data_path)
    df.drop(columns=["Record_ID", "Bidder_ID", "Auction_ID"], inplace=True)
    categorical_features = [2, 8]
    y = df["Class"]
    X = df.drop(columns="Class")
    return {
        "X": X.to_numpy(),
        "y": y.to_numpy(),
        "regression": False,
        "categorical_features": categorical_features,
    }


def prepare_wine_quality(data_folder=DATASET_FOLDER):
    """Wine Quality Data Set"""
    data_path = os.path.join(data_folder, "winequality-white.csv")
    df = pd.read_csv(data_path, delimiter=";")
    y = df["quality"]
    X = df.drop(columns="quality")
    return {
        "X": X.to_numpy(),
        "y": y.to_numpy(),
        "regression": True,
        "categorical_features": [],
    }


dataset_loaders = {
    "ai4i2020": prepare_ai4i2020,
    "airfoil": prepare_airfoil,
    "bank_marketing": prepare_bank_marketing,
    "census_income": prepare_census_income,
    "power_plant": prepare_power_plant,
    "concrete": prepare_concrete,
    "default_credit_card": prepare_default_credit_card,
    "diabetic_retinopathy": prepare_diabetic_retinopathy,
    "electrical_grid": prepare_electrical_grid,
    "gas_turbine": prepare_gas_turbine,
    "mini_boone": prepare_mini_boone,
    "online_shoppers": prepare_online_shoppers,
    "seismic_bumps": prepare_seismic_bumps,
    "shill_bidding": prepare_shill_bidding,
    "wine_quality": prepare_wine_quality,
}
