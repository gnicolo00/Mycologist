import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Lettura del dataset contenente le informazioni relative ai funghi, utilizzando la libreria 'os' per garantire una
# corretta funzionalità in ogni sistema operativo
dataset_path = os.path.join("..", "datasets", "mushrooms.csv")
dataset = pd.read_csv(dataset_path)
print(dataset, end="\n\n---------------------------------------------\n\n")


# Conteggio delle righe del dataset
print(f"Dataset samples count: {len(dataset)-1}", end="\n\n---------------------------------------------\n\n")


# Analisi delle colonne del dataset
print("Dataset columns:")
for index, column in enumerate(dataset.columns, start=1):
    print(f"{index}. {column}")
print("\n---------------------------------------------\n")


# Analisi della classe da predire
print(f"Class to predict: {dataset['class'].unique()}", end="\n\n---------------------------------------------\n\n")


# Ricerca di eventuali valori mancanti
print("Missing values:")
# Poiché, in questo dataset, i valori mancanti sono riportati come punti interrogativi piuttosto che valori nulli, si
# esegue una modifica per modificare i primi con gli ultimi ('inplace' permette di modificare il dataset stesso)
dataset.replace('?', np.nan, inplace=True)
# Creazione di un dataset booleano della stessa forma di 'dataset', con la differenza che ogni elemento è True se il
# valore corrispondente di 'dataset' è nullo, altrimenti False
boolean_dataset = dataset.isna()
# Calcolo della somma di tutti i valori nulli per ogni colonna (poiché True è considerato come 1 e False come 0)
missing_values_count = boolean_dataset.sum()
print(missing_values_count, end="\n\n---------------------------------------------\n\n")


# Creazione di due dataset differenti: uno in cui tutti i funghi sono commestibili e l'altro in cui tutti sono velenosi
dataset.replace(np.nan, '?', inplace=True)  # I valori nulli non permettono il corretto funzionamento dei plot
dataset_edible = dataset[dataset['class'] == 'e']
dataset_poisonous = dataset[dataset['class'] == 'p']
# Creazione degli istogrammi relativi alla distribuzione delle classi rispetto a ciascuna caratteristica del dataset
for column in dataset.columns[1:]:
    # Inizializzazione di un array contenente tutti i valori che la colonna può assumere
    unique_values = dataset[column].unique()
    # Inizializzazione di un array contenente valori da 0 al numero di valori della colonna, che corrispondera al numero
    # dei bins (barre verticali) contenuti nell'istogramma. Si sottrae 0.5 per centrare il bin rispetto al valore
    bins = np.arange(len(unique_values) + 1) - 0.5

    # Creazione dell'istogramma
    plt.hist([dataset_edible[column], dataset_poisonous[column]], color=["#E1CDC2", "#D50630"], edgecolor='black',
             alpha=0.8, stacked=True, rwidth=0.5, bins=bins)

    # Creazione dell'etichetta principale sull'asse x dell'istogramma
    plt.xlabel(column.replace('-', ' ').capitalize())
    # Vengono create tante etichette quanto il numero di valori che la colonna può assumere, e ogni etichetta viene
    # posizionata alla posizione corrispondente
    plt.xticks(range(len(unique_values)), unique_values)
    # Creazione dell'etichetta principale sull'asse y dell'istogramma
    plt.ylabel("Frequency")
    # Creazione della legend dell'istogramma
    plt.legend(["Edible", "Poisonous"])
    # Visualizzazione dell'istogramma
    plt.show()
# Reinserimento dei valori nulli al posto dei punti interrogativi
dataset.replace('?', np.nan, inplace=True)  # I valori nulli non permettono il corretto funzionamento dei plot
