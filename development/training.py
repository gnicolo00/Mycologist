import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
import os

# Recupero del dataset contenente le informazioni relative ai funghi, utilizzando la libreria 'os' per garantire una
# corretta funzionalità in ogni sistema operativo
dataset = pd.read_csv(os.path.join("..", "datasets", "mushrooms_numbers.csv"), dtype="Int64")


# Ottenimento delle labels
y = dataset["poisonous"]
# Ottenimento del dataset senza labels
X = dataset.drop("poisonous", axis=1)


# Data splitting (80-20) casuale per ottenere il dataset di training e il dataset di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Dictionary contenente i diversi classificatori
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'Linear Regression': LinearRegression()
}
# Dictionaries contenente le diverse metriche di valutazione
accuracy = {'Decision Tree': [], 'Random Forest': [], 'Naive Bayes': [], 'Linear Regression': []}
precision = {'Decision Tree': [], 'Random Forest': [], 'Naive Bayes': [], 'Linear Regression': []}
recall = {'Decision Tree': [], 'Random Forest': [], 'Naive Bayes': [], 'Linear Regression': []}
f1_score = {'Decision Tree': [], 'Random Forest': [], 'Naive Bayes': [], 'Linear Regression': []}
