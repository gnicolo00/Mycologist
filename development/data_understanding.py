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
