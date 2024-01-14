from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scikit-learn.preprocessing import StandardScaler
from scikit-learn.impute import SimpleImputer
from scikit-learn.compose import ColumnTransformer

from scikit-learn.linear_model import LogisticRegression
from scikit-learn.multioutput import MultiOutputClassifier

from scikit-learn.pipeline import Pipeline

from scikit-learn.model_selection import train_test_split

from scikit-learn.metrics import roc_curve, roc_auc_score

RANDOM_SEED = 6    # Set a random seed for reproducibility!

pd.set_option("display.max_columns", 100)

DATA_PATH = Path.cwd().parent / "Flu-Shot-Learning" / "Data"

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

features_df.dtypes

print("labels_df.shape", labels_df.shape)
labels_df.head()

np.testing.assert_array_equal(features_df.index.values, labels_df.index.values)

fig, ax = plt.subplots(2, 1, sharex=True)

n_obs = labels_df.shape[0]

(labels_df['h1n1_vaccine']
    .value_counts()
    .div(n_obs)
    .plot.barh(title="Proportion of H1N1 Vaccine", ax=ax[0])
)

ax[0].set_ylabel("h1n1_vaccine")

(labels_df['seasonal_vaccine']
    .value_counts()
    .div(n_obs)
    .plot.barh(title="Proportion of Seasonal Vaccine", ax=ax[1])
)

ax[1].set_ylabel("seasonal_vaccine")

fig.tight_layout()

pd.crosstab(
    labels_df["h1n1_vaccine"], 
    labels_df["seasonal_vaccine"], 
    margins=True,
    normalize=True
)

(labels_df["h1n1_vaccine"]
     .corr(labels_df["seasonal_vaccine"], method="pearson")
)

joined_df = features_df.join(labels_df)
print("joined_df.shape", joined_df.shape)
joined_df.head()

counts = (joined_df[['h1n1_concern', 'h1n1_vaccine']]
              .groupby(['h1n1_concern', 'h1n1_vaccine'])
              .size()
              .unstack('h1n1_vaccine')
         )
print(counts)

h1n1_concern_counts = counts.sum(axis='columns')
print("h1n1_concern_counts",h1n1_concern_counts)

props = counts.div(h1n1_concern_counts, axis='index')
print("props",props)

ax = props.plot.barh(title="H1N1 Concern x Vaccine",stacked=True)
ax.invert_yaxis()
ax.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5),
    title='h1n1_vaccine'
)

def vaccination_rate_plot(col, target, data, ax=None):
    counts = (joined_df[[target, col]]
                  .groupby([target, col])
                  .size()
                  .unstack(target)
             )
    group_counts = counts.sum(axis='columns')
    props = counts.div(group_counts, axis='index')

    props.plot(kind="barh", stacked=True, ax=ax)
    ax.invert_yaxis()
    ax.legend().remove()


cols_to_plot = [
    'h1n1_concern',
    'h1n1_knowledge',
    'opinion_h1n1_vacc_effective',
    'sex',
    'age_group',
    'race',
]

fig, ax = plt.subplots(
    len(cols_to_plot), 2, figsize=(9,len(cols_to_plot)*2.5)
)
for idx, col in enumerate(cols_to_plot):
    vaccination_rate_plot(
        col, 'h1n1_vaccine', joined_df, ax=ax[idx, 0]
    )
    vaccination_rate_plot(
        col, 'seasonal_vaccine', joined_df, ax=ax[idx, 1]
    )
    
ax[0, 0].legend(
    loc='lower center', bbox_to_anchor=(0.5, 1.05), title='h1n1_vaccine'
)
ax[0, 1].legend(
    loc='lower center', bbox_to_anchor=(0.5, 1.05), title='seasonal_vaccine'
)
fig.tight_layout()

plt.show()