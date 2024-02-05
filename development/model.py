import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

    save_evaluation_graphs(y_test, testing_predictions, model_name)


def save_evaluation_graphs(real_values, predictions, model_name):
    # Ottenimento delle iniziali del modello (per il nome dei file)
    model_initials = ''.join(word[0].lower() for word in model_name.split())

    # Chiusura di tutte le figure "aperte" per evitare sovrapposizioni e visualizzazioni distorte
    plt.close("all")
    # Creazione della matrice di confusione relativa al modello Naive Bayes
    matrix = confusion_matrix(y_true=real_values, y_pred=predictions)
    # Definizione dei colori da utilizzare nella matrice con relativa color map e color matrix
    color_map = ListedColormap('white', name='colormap_list')
    color_matrix = [['#E1CDC2', '#D50630'], ['#D50630', '#E1CDC2']]
    color_text_matrix = [['black', 'white'], ['white', 'black']]
    # Visualizzazione della matrice di confusione
    plt.imshow(matrix, cmap=color_map, origin='upper')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Definizione dei dettagli in merito a colore e testo per le celle della matrice
            plt.text(j, i, str(matrix[i, j]), color=color_text_matrix[i][j])
            plt.fill_between([j - 0.5, j + 0.5], i - 0.5, i + 0.5, color=color_matrix[i][j], alpha=1)
    # Definizione dei valori e delle etichette presenti sull'asse x e sull'asse y della matrice di confusione
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # Salvataggio della matrice di confusione
    plt.savefig(os.path.join("..", "plots", model_initials + "_confusion_matrix.png"), format="png")

    # Chiusura di tutte le figure "aperte" per evitare sovrapposizioni e visualizzazioni distorte
    plt.close("all")
    # Calcolo del tasso di falsi positivi, veri positivi e le soglie
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, predictions)
    # Disegno della linea di riferimento
    plt.plot([0, 1], [0, 1], 'k--')
    # Disegno della ROC Curve, etichettandola con il valore della AUC (più vicino è a 1, migliore è il modello)
    plt.plot(false_positive_rate, true_positive_rate,
             label='AUC = {:.4f})'.format(auc(false_positive_rate, true_positive_rate)), color="#D50630")
    # Definizione delle etichette presenti sull'asse x e sull'asse y della ROC Curve
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')

    # Salvataggio della ROC Curve
    plt.savefig(os.path.join("..", "plots", model_initials + "_roc_curve.png"), format="png")


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
