import os
import pandas as pd
import gdown

def download_data(output_dir='data'):
    """
    Downloads the Yelp review dataset if it doesn't exist.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    target_dir = os.path.join(output_dir, 'yelp_review')
    if os.path.exists(target_dir):
        print(f"Data directory '{target_dir}' already exists. Skipping download.")
        return

    print("Downloading data...")
    url = 'https://drive.google.com/drive/folders/1zF5s4KRMxpr3OvUY8o_d0xv_YlaZSCsj?usp=drive_link'
    try:
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
        print("Download completed.")
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please check your internet connection or the Google Drive link.")

def load_data(data_dir='data/yelp_review'):
    """
    Loads train, validation, and test datasets.
    """
    try:
        train_path = os.path.join(data_dir, 'train.tsv')
        val_path = os.path.join(data_dir, 'val.tsv')
        test_path = os.path.join(data_dir, 'test.tsv')

        print(f"Loading data from {data_dir}...")
        train_set = pd.read_csv(train_path, sep='\t')
        dev_set = pd.read_csv(val_path, sep='\t')
        test_set = pd.read_csv(test_path, sep='\t')
        
        print(f"Train size: {len(train_set)}")
        print(f"Dev size: {len(dev_set)}")
        print(f"Test size: {len(test_set)}")

        return train_set, dev_set, test_set
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise
