import pandas as pd
import os

# Recupero del dataset contenente le informazioni relative ai funghi, utilizzando la libreria 'os' per garantire una
# corretta funzionalità in ogni sistema operativo
dataset_path = os.path.join("..", "datasets", "mushrooms_numbers.csv")
dataset = pd.read_csv(dataset_path)

# Eliminazione delle colonne inadeguate alla soluzione
dataset1 = dataset.drop(["odor", "gill-attachment", "veil-type", "veil-color"], axis=1)
# Salvataggio delle modifiche
dataset1_path = os.path.join("..", "datasets", "mushrooms_deleted.csv")
dataset1.to_csv(dataset1_path, index=False)
