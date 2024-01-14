# Importando bibliotecas necessárias
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importando módulos do scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# Definindo uma semente aleatória para garantir reprodutibilidade
RANDOM_SEED = 6 

# Configurando opções de exibição do Pandas
pd.set_option("display.max_columns", 100)

# Definindo o caminho para os dados
DATA_PATH = Path.cwd().parent / "Flu-Shot-Learning" / "Data"

# Carregando os dados de treinamento
features_df = pd.read_csv(
    DATA_PATH / "training_set_features.csv",
    index_col="respondent_id"
)

labels_df = pd.read_csv(
    DATA_PATH / "training_set_labels.csv",
    index_col="respondent_id"
)

# Verificando se os índices dos dataframes são iguais
np.testing.assert_array_equal(features_df.index.values, labels_df.index.values)

# Juntando os dataframes de features e labels
joined_df = features_df.join(labels_df)

# Função para plotar a taxa de vacinação
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

# Selecionando colunas numéricas
numeric_cols = features_df.columns[features_df.dtypes != "object"].values

# Criando etapas de pré-processamento para colunas numéricas
numeric_preprocessing_steps = Pipeline([
    ('standard_scaler', StandardScaler()),  # Padronização dos dados
    ('simple_imputer', SimpleImputer(strategy='median'))  # Imputação de valores ausentes usando a mediana
])

# Criando um transformador de colunas numéricas
preprocessor = ColumnTransformer(
    transformers = [
        ("numeric", numeric_preprocessing_steps, numeric_cols)
    ],
    remainder = "drop"
)

# Criando o classificador multi-output (para prever múltiplos rótulos)
estimators = MultiOutputClassifier(
    estimator=LogisticRegression(penalty="l2", C=1)  # Regressão Logística com regularização L2
)

# Criando o pipeline completo
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimators", estimators),
])

# Dividindo os dados em conjuntos de treinamento e avaliação
X_train, X_eval, y_train, y_eval = train_test_split(
    features_df,
    labels_df,
    test_size=0.33,
    shuffle=True,
    stratify=labels_df,
    random_state=RANDOM_SEED
)

# Treinando o modelo no conjunto de treinamento
full_pipeline.fit(X_train, y_train)

# Fazendo previsões no conjunto de avaliação
preds = full_pipeline.predict_proba(X_eval)

# Criando um dataframe para as previsões
y_preds = pd.DataFrame(
    {
        "h1n1_vaccine": preds[0][:, 1],
        "seasonal_vaccine": preds[1][:, 1],
    },
    index = y_eval.index
)

# Função para plotar a curva ROC
def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title(
        f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}"
    )

# Avaliando o modelo usando a curva ROC no conjunto de avaliação
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_roc(y_eval['h1n1_vaccine'], y_preds['h1n1_vaccine'], 'h1n1_vaccine', ax=ax[0])
plot_roc(y_eval['seasonal_vaccine'], y_preds['seasonal_vaccine'], 'seasonal_vaccine', ax=ax[1])
fig.tight_layout()

# Treinando o modelo no conjunto completo de treinamento
full_pipeline.fit(features_df, labels_df)

# Carregando os dados de teste
test_features_df = pd.read_csv(DATA_PATH / "test_set_features.csv", index_col="respondent_id")

# Fazendo previsões no conjunto de teste
test_probas = full_pipeline.predict_proba(test_features_df)

# Carregando o formato de submissão
submission_df = pd.read_csv(DATA_PATH / "submission_format.csv", index_col="respondent_id")

# Verificando se os índices dos dados de teste e submissão são iguais
np.testing.assert_array_equal(test_features_df.index.values, submission_df.index.values)

# Preenchendo as colunas de vacinação no dataframe de submissão com as probabilidades previstas
submission_df["h1n1_vaccine"] = test_probas[0][:, 1]
submission_df["seasonal_vaccine"] = test_probas[1][:, 1] 

# Salvando a submissão em um arquivo CSV
submission_df.to_csv('my_submission.csv', index=True)
