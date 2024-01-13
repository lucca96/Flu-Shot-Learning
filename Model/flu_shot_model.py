from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns",100)

DATA_PATH = Path.cwd().parent / "Flu-Shot-Learning" / "Data"  #creating path shortcut

features_df = pd.read_csv(
    DATA_PATH / "training_set_features.csv",
    index_col="respondent_id"
)
labels_df = pd.read_csv(
    DATA_PATH / "training_set_labels.csv",
    index_col="respondent_id"
)

print("features_df.shape", features_df.shape)
features_df.head()