import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import os

# Lettura del dataset contenente le informazioni relative ai funghi, utilizzando la libreria 'os' per garantire una
# corretta funzionalità in ogni sistema operativo
dataset_path = os.path.join("..", "datasets", "mushrooms_initial.csv")
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
# esegue una modifica per modificare i primi con gli ultimi. Le modifiche sono infine trasferite in un altro dataset
dataset1 = dataset.replace('?', np.nan)
# Creazione di un dataset booleano della stessa forma di 'dataset1', con la differenza che ogni elemento è True se il
# valore corrispondente di 'dataset1' è nullo, altrimenti False
boolean_dataset = dataset1.isna()
# Calcolo della somma di tutti i valori nulli per ogni colonna (poiché True è considerato come 1 e False come 0)
missing_values_count = boolean_dataset.sum()
print(missing_values_count, end="\n\n---------------------------------------------\n\n")
# Salvataggio delle modifiche
dataset1_path = os.path.join("..", "datasets", "mushrooms_nan")
dataset1.to_csv(dataset1_path, index=False)


# Creazione di due dataset differenti: uno in cui tutti i funghi sono commestibili e l'altro in cui tutti sono velenosi
dataset_edible = dataset[dataset['class'] == 'e']
dataset_poisonous = dataset[dataset['class'] == 'p']
# Creazione degli istogrammi relativi alla distribuzione delle classi rispetto a ciascuna caratteristica del dataset
for column in dataset.columns[1:]:
    # Inizializzazione di un array contenente tutti i valori che la colonna può assumere
    unique_values = dataset[column].unique()
    # Inizializzazione di un array contenente valori da 0 al numero di valori della colonna, che corrisponderà al numero
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


# Conversione delle variabili categoriche in variabili numeriche (encoding), necessario per creare la heatmap
dataset2 = pd.DataFrame(dataset1)
for column in dataset2.columns:
    # Salvataggio della colonna prima dell'operazione di encoding
    column_before = pd.unique(dataset2[column])
    # Label encoding
    dataset2[column] = LabelEncoder().fit_transform(dataset1[column])
    # Salvataggio della colonna dopo l'operazione di encoding
    column_after = pd.unique(dataset2[column])

    # Visualizzazione dei risultati prima e dopo l'operazione di encoding, così da capire a quale valore corrisponde un
    # determinato numero nella colonna
    print(f"{column}")
    for i in range(len(pd.unique(dataset2[column]))):
        print(f"{column_before[i]} → {column_after[i]}", end=", " if i < len(column_before) - 1 else "")
    print("\n")

# Cambio del nome della colonna relativa alla variabile dipendente
dataset2 = dataset2.rename(columns={'class': 'poisonous'})
# Salvataggio delle modifiche
dataset2_path = os.path.join("..", "datasets", "mushrooms_numbers")
dataset2.to_csv(dataset2_path, index=False)


# Calcolo delle correlazioni tra le features del dataset
correlations = dataset2[dataset2.columns[1:]].corr()
# Risoluzione dell'immagine
fig, ax = plt.subplots(figsize=(8, 7))
# Posizionamento della heatmap all'interno dell'immagine
fig.subplots_adjust(left=0.15, right=1, top=1, bottom=0.1)
# Creazione della heatmap per analizzare le correlazioni tra le features del dataset
heatmap = sns.heatmap(correlations, cmap="hot", vmin=-1, vmax=1, annot=True, annot_kws={'fontsize': 5},
                      cbar_kws={"shrink": 0.7}, square=True, ax=ax)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=5)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=5)
plt.show()
