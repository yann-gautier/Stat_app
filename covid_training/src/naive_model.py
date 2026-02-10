import pandas as pd
import bnlearn as bn
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings

# Suppression des alertes
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "../data/combined_covid_data.csv"
OUTPUT_IMG = SCRIPT_DIR / "../data/dag_modele_naif_bnlearn.png"

TARGET_VARS = [
    "Gender", "Ethnicity", "Age_at_ICI_start",
    "Simplified_Stage", "ECOG", "CNS_disease", "Previous_history_of_malignancy_at_ICI_start",
    "Vaccine100", "Steroid_win_1_month_of_Vaccine", "Concurrent_Chemo", "BRAF",
    "PFS_", "PFS_Code",
    "OS_months", "OS_event"
]

# -------------------------------------------------------------------
# 2. CHARGEMENT
# -------------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Erreur : {DATA_PATH} introuvable.")

print(f"--- Démarrage de l'analyse naïve ---")
df = pd.read_csv(DATA_PATH)
existing_vars = [col for col in TARGET_VARS if col in df.columns]
df_clean = df[existing_vars].copy().dropna()

# Encodage numérique
for col in df_clean.columns:
    if df_clean[col].dtype == 'object' or df_clean[col].dtype == 'bool':
        df_clean[col] = df_clean[col].astype('category').cat.codes

# -------------------------------------------------------------------
# 3. APPRENTISSAGE
# -------------------------------------------------------------------
print(f"Apprentissage du DAG (Hill Climbing)...")
model_naive = bn.structure_learning.fit(df_clean, methodtype='hc', scoretype='bic', verbose=0)
adj_mat = model_naive['adjmat']
n_edges = np.sum(adj_mat.values)
print(f"Terminé. Nombre d'arêtes trouvées : {n_edges}")

# -------------------------------------------------------------------
# 4. VISUALISATION ROBUSTE
# -------------------------------------------------------------------
print(f"Génération de l'image via NetworkX...")

G = nx.from_pandas_adjacency(adj_mat, create_using=nx.DiGraph)

plt.figure(figsize=(14, 10))

# Disposition
pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)

# Définition de la taille des nœuds (variable pour être sûr que ça matche)
MY_NODE_SIZE = 2500

# 1. Dessiner les nœuds
nx.draw_networkx_nodes(G, pos, 
                       node_size=MY_NODE_SIZE, 
                       node_color='#e0f7fa', 
                       edgecolors='#006064')

# 2. Dessiner les labels
nx.draw_networkx_labels(G, pos, 
                        font_size=8, 
                        font_weight='bold', 
                        font_family='sans-serif')

# 3. Dessiner les arêtes (CORRECTION MAJEURE ICI)
nx.draw_networkx_edges(G, pos, 
                       node_size=MY_NODE_SIZE,  # <--- INDISPENSABLE : dit à la flèche de s'arrêter au bord
                       edge_color='#006064', 
                       width=2.0, 
                       arrowsize=25, 
                       arrowstyle='-|>',        # Style de flèche "Triangle plein"
                       connectionstyle="arc3,rad=0.1", 
                       min_target_margin=15,    # Marge de sécurité supplémentaire
                       alpha=0.8)

plt.title(f"DAG Naïf (Hill Climbing) - {n_edges} liens détectés", fontsize=16, fontweight='bold')
plt.axis('off') 
plt.tight_layout()

print(f"Sauvegarde dans : {OUTPUT_IMG.name}")
plt.savefig(OUTPUT_IMG, bbox_inches='tight', dpi=300)
plt.close()

# -------------------------------------------------------------------
# 5. ANALYSE
# -------------------------------------------------------------------
print("\n--- Analyse des liens ---")
if 'Vaccine100' in adj_mat.columns and 'OS_event' in adj_mat.columns:
    print(f"Vaccin -> Survie : {adj_mat.loc['Vaccine100', 'OS_event']}")
    print(f"Survie -> Vaccin : {adj_mat.loc['OS_event', 'Vaccine100']}")
else:
    print("Variables Vaccin/Survie absentes du graphe.")