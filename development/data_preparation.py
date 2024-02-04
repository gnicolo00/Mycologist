import pandas as pd
import os

# Recupero del dataset contenente le informazioni relative ai funghi, utilizzando la libreria 'os' per garantire una
# corretta funzionalità in ogni sistema operativo
dataset = pd.read_csv(os.path.join("..", "datasets", "mushrooms_numbers.csv"), dtype="Int64")


# Eliminazione delle colonne inadeguate alla soluzione
dataset1 = dataset.drop(["odor", "gill-attachment", "veil-type", "veil-color"], axis=1)
# Salvataggio delle modifiche
dataset1_path = os.path.join("..", "datasets", "mushrooms_deleted.csv")
dataset1.to_csv(dataset1_path, index=False)


# Imputazione con valore unico sulla colonna con valori mancanti, utilizzando la moda
dataset2 = pd.DataFrame(dataset1)
dataset2["stalk-root"] = dataset1["stalk-root"].fillna(dataset1["stalk-root"].mode()[0])
# Salvataggio delle modifiche
dataset2_path = os.path.join("..", "datasets", "mushrooms_imputed.csv")
dataset2.to_csv(dataset2_path, index=False)
