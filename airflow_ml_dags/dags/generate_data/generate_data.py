from hypothesis import strategies
from hypothesis.extra.pandas import data_frames, column, range_indexes
import os

def generate_data(year: str,
                       month: str,
                       day: str,
                       hour: str,
                       output_dir: str,
                       ) -> None:
    df = data_frames(
        index=range_indexes(min_size=100),
        columns=[
            column('chol', dtype=float, elements=strategies.floats(min_value=100, max_value=900)),
            column('thalach', dtype=int, elements=strategies.integers(min_value=30, max_value=250)),
            column('oldpeak', dtype=float, elements=strategies.floats(min_value=0, max_value=7)),
            column('trestbps', dtype=int, elements=strategies.integers(min_value=80, max_value=250)),
            column('fbs', dtype=int, elements=strategies.integers(min_value=0, max_value=1)),
            column('age', dtype=int, elements=strategies.integers(min_value=18, max_value=120)),
            column('sex', dtype=int, elements=strategies.integers(min_value=0, max_value=1)),
            column('cp', dtype=int, elements=strategies.integers(min_value=0, max_value=3)),
            column('restecg', dtype=int, elements=strategies.integers(min_value=0, max_value=2)),
            column('exang', dtype=int, elements=strategies.integers(min_value=0, max_value=1)),
            column('slope', dtype=int, elements=strategies.integers(min_value=0, max_value=2)),
            column('ca', dtype=int, elements=strategies.integers(min_value=0, max_value=4)),
            column('thal', dtype=int, elements=strategies.integers(min_value=1, max_value=3)),
            column('target', dtype=int, elements=strategies.integers(min_value=0, max_value=1)),
        ]).example()

    directory = f"{output_dir}/{year}/{month}/{day}/{hour}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(f"{directory}/data.csv", 'w') as f:
        df.drop(columns=['target']).to_csv(f)
    with open(f"{directory}/target.csv", 'w') as f:
        df[['target']].to_csv(f)
