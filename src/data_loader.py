import os
import glob
import pandas as pd

def load_raw_data(data_dir):
    path = os.path.join(data_dir, "*.csv")
    all_files = glob.glob(path)
    df_raw = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    print("Raw rows:", len(df_raw))
    return df_raw