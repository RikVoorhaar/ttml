# %%
dataset_data = {
    "ai4i2020": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
    },
    "airfoil": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
    },
    "bank_marketing": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
        "rename": "bank_marketing.csv",
        "unzip": "bank-full.csv",
    },
    "census_income": {
        "url": "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "rename": "census_income.data",
    },
    "power_plant": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
        "unzip": "CCPP/Folds5x2_pp.xlsx",
        "rename": "combinedCyclePowerPlant.xlsx",
    },
    "concrete": {
        "url": "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
    },
    "default_credit_card": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        "rename": "default_of_credit_card_clients.xls",
    },
    "diabetic_retinopathy": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff",
        "rename": "diabetic_retinopathy_debrecen.arff",
    },
    "electrical_grid": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv",
        "rename": "electric_grid_stability.csv",
    },
    "gas_turbine": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00551/pp_gas_emission.zip",
        "rename": "gas_turbine_emission.zip",
    },
    # "vehicle_coupon": {
    #     "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00603/in-vehicle-coupon-recommendation.csv"
    # },
    "mini_boone": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt"
    },
    "online_shoppers": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    },
    # "bankruptcy": {
    #     "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip"
    # },
    "seismic_bumps": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff"
    },
    "shill_bidding": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00562/Shill%20Bidding%20Dataset.csv",
        "rename": "Shill_Bidding_Dataset.csv",
    },
    "wine_quality": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    },
}
# %%
import zipfile
from tqdm import tqdm
import urllib.request
import os.path
import os
from zipfile import ZipFile

DATA_FOLDER = "data"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to
        )


def download_dataset(url=None, rename=None, unzip=None):
    if (rename is None) or (unzip is not None):
        file_name = url.split("/")[-1]
    else:
        file_name = rename
    file_path = os.path.join(DATA_FOLDER, file_name)
    print(file_path)
    if unzip is not None:
        print("Unzipping file...")
        if rename is not None:
            unzip_rename = rename
        else:
            unzip_rename = unzip
        unzip_rename_file_path = os.path.join(DATA_FOLDER, unzip_rename)
        unzip_file_path = os.path.join(DATA_FOLDER, unzip)
        if os.path.isfile(unzip_rename_file_path):
            print("Dataset already downloaded, skipping...")
            return

    if os.path.isfile(file_path):
        print("Dataset already downloaded, skipping...")
        return
    else:
        download_url(url, file_path)

    if unzip is not None:
        print(file_name)
        with ZipFile(file_path) as data_zip:
            data_zip.extract(unzip, DATA_FOLDER)
        os.rename(unzip_file_path, unzip_rename_file_path)
        os.remove(file_path)

    print("Done!")


def test_datasets():
    from load_data import dataset_loaders

    for name, loader in dataset_loaders.items():
        print(f"Testing dataset '{name}'")

        loader()


if __name__ == "__main__":

    try:
        os.mkdir(DATA_FOLDER)
    except FileExistsError:
        pass

    for dataset_name, dataset in dataset_data.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 80)
        download_dataset(**dataset)

    print("Testing if all datasets loaded correctly...")
    print("-" * 80)
    test_datasets()

    print("\nAll datasets loaded correctly!")
