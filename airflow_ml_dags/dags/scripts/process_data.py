import pandas as pd
from sklearn.model_selection import train_test_split
import os


def process_data(year: str,
                 month: str,
                 day: str,
                 raw_data_path: str,
                 output_dir: str
                 ) -> None:
    full_raw_data_dir = f"{raw_data_path}/{year}/{month}/{day}"
    data = pd.read_csv(f"{full_raw_data_dir}/data.csv")
    target = pd.read_csv(f"{full_raw_data_dir}/target.csv")
    df = pd.merge(data, target, left_index=True, right_index=True, how="inner")
    df = df[df['chol'] < 700]
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=2)
    directory = f"{output_dir}/{year}/{month}/{day}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(f"{directory}/train.csv", 'w') as f:
        train_data.to_csv(f, index=False)
    with open(f"{directory}/val.csv", 'w') as f:
        val_data.to_csv(f, index=False)
