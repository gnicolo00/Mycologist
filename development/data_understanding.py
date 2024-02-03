import pandas as pd
import matplotlib.pyplot as plt
import os

# Lettura del dataset contenente le informazioni relative ai funghi, utilizzando la libreria 'os' per garantire una
# corretta funzionalità in ogni sistema operativo
dataset_path = os.path.join("..", "datasets", "mushrooms.csv")
dataset = pd.read_csv(dataset_path)
print(dataset, end="\n\n---------------------------------------------\n\n")

# Conteggio delle righe del dataset
print(f"Dataset samples count: {len(dataset)-1}", end="\n\n---------------------------------------------\n\n")
