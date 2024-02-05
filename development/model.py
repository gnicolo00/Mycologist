import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import os

# Recupero del dataset contenente le informazioni relative ai funghi, utilizzando la libreria 'os' per garantire una
# corretta funzionalità in ogni sistema operativo
dataset = pd.read_csv(os.path.join("..", "datasets", "mushrooms_imputed.csv"), dtype="Int64")


# Ottenimento delle labels
y = dataset["poisonous"]
# Ottenimento del dataset senza labels
X = dataset.drop("poisonous", axis=1)


# Data splitting (80-20) casuale per ottenere il dataset di training e il dataset di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Inizializzazione dei diversi classificatori
nb = MultinomialNB()
lr = LogisticRegression(max_iter=200)
