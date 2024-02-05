import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    # Addestramento del modello
    model.fit(X_train, y_train)

    # Previsioni del modello sui dati di training
    training_predictions = model.predict(X_train)
    # Previsioni del modello sui dati di testing
    testing_predictions = model.predict(X_test)

    # Calcolo delle metriche sui dati di training
    accuracy_training = accuracy_score(y_true=y_train, y_pred=training_predictions)
    precision_training = precision_score(y_true=y_train, y_pred=training_predictions)
    recall_training = recall_score(y_true=y_train, y_pred=training_predictions)
    f1_training = f1_score(y_true=y_train, y_pred=training_predictions)
    # Calcolo delle metriche sui dati di testing
    accuracy_testing = accuracy_score(y_true=y_test, y_pred=testing_predictions)
    precision_testing = precision_score(y_true=y_test, y_pred=testing_predictions)
    recall_testing = recall_score(y_true=y_test, y_pred=testing_predictions)
    f1_testing = f1_score(y_true=y_test, y_pred=testing_predictions)

    # Visualizzazione delle metriche
    print(f"{model_name.upper()} (TRAINING):\n"
          f"Accuracy: {round(accuracy_training, 2)}\n"
          f"Precision: {round(precision_training, 2)}\n"
          f"Recall: {round(recall_training, 2)}\n"
          f"F1 Score: {round(f1_training, 2)}")
    print(f"{model_name.upper()} (TESTING):\n"
          f"Accuracy: {round(accuracy_testing, 2)}\n"
          f"Precision: {round(precision_testing, 2)}\n"
          f"Recall: {round(recall_testing, 2)}\n"
          f"F1 Score: {round(f1_testing, 2)}")


# Recupero del dataset contenente le informazioni relative ai funghi, utilizzando la libreria 'os' per garantire una
# corretta funzionalità in ogni sistema operativo
dataset = pd.read_csv(os.path.join("..", "datasets", "mushrooms_imputed.csv"), dtype="Int64")


# Ottenimento delle labels
y = dataset["poisonous"]
# Ottenimento del dataset senza labels
X = dataset.drop("poisonous", axis=1)


# Data splitting (80-20) casuale per ottenere il dataset di training e il dataset di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Inizializzazione dei modelli
nb = MultinomialNB()
lr = LogisticRegression(max_iter=200)


# Addestramento dei modelli e valutazione delle performance
train_and_evaluate(nb, "Naive Bayes", X_train, X_test, y_train, y_test)
train_and_evaluate(lr, "Logistic Regression", X_train, X_test, y_train, y_test)
