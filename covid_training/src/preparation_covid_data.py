import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------
# 1. Paramètres de base
# -------------------------------------------------------------------
# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "../ressources/covid_data.xlsx"  # chemin fourni par ton environnement

SHEETS_TO_LOAD = [
    "1a Raw Data",
    "1b Raw Data",
    "1c Raw Data",
    "1d Raw Data",
]

OUTPUT_PATH = SCRIPT_DIR / "../ressources/combined_covid_data.csv"

# Options de gestion des valeurs manquantes :
# "keep" : garde toutes les lignes et colonnes (par défaut)
# "drop_rows" : supprime les lignes avec au moins une valeur manquante
# "drop_cols" : supprime les colonnes avec au moins une valeur manquante
MISSING_DATA_STRATEGY = "drop_rows"  # Options: "keep", "drop_rows", "drop_cols"

# Seuil pour drop_cols : supprimer les colonnes avec plus de X% de valeurs manquantes
MISSING_THRESHOLD = 0  # 0.5 = 50% de valeurs manquantes


# -------------------------------------------------------------------
# 2. Petites fonctions utilitaires
# -------------------------------------------------------------------

def get_common_columns(excel_path: Path, sheets: list[str]) -> list[str]:
    """
    Identifie les colonnes communes à toutes les feuilles.
    """
    xls = pd.ExcelFile(excel_path)
    sheets_columns = []
    
    for sheet in sheets:
        if sheet not in xls.sheet_names:
            print(f"⚠️ Feuille {sheet} absente du fichier, ignorée.")
            continue
        
        df = pd.read_excel(excel_path, sheet_name=sheet, nrows=0)  # Charge seulement les colonnes
        sheets_columns.append(set(df.columns))
    
    if not sheets_columns:
        raise ValueError("Aucune feuille trouvée.")
    
    # Intersection de toutes les colonnes
    common_columns = set.intersection(*sheets_columns)
    
    print(f"✅ {len(common_columns)} colonnes communes trouvées sur {len(sheets)} feuilles")
    
    return sorted(common_columns)


def standardize_columns(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Harmonise les noms de colonnes clés entre les feuilles 1a–1d.
    On ne touche qu'aux colonnes qui changent de nom selon la feuille.
    On ajoute aussi une colonne 'source_sheet' pour garder la trace.
    """
    df = df.copy()
    df["source_sheet"] = sheet_name

    # mapping des noms de colonnes spécifiques vers un nom standard
    rename_map = {
        # OS (overall survival)
        "OS_from_relevant_ICI_start,_months": "OS_months",
        "OS_from_relevant_ICI_start_months": "OS_months",
        "OS_Months": "OS_months",

        # code évènement OS
        "OS_Code_(1=death, 0=censored)": "OS_event",
        "OS_Code_1=death": "OS_event",
        "OS_Code": "OS_event",
        "OS_Code_": "OS_event",

        # groupe vaccin
        "Group_(1=vaccine)": "Vaccine100",
        "Group_1=vaccine": "Vaccine100",
    }

    df = df.rename(columns=rename_map)

    # s'assurer que certaines colonnes existent (parfois absentes dans une feuille)
    if "OS_months" not in df.columns:
        df["OS_months"] = np.nan
    if "OS_event" not in df.columns:
        df["OS_event"] = np.nan
    if "Vaccine100" not in df.columns:
        df["Vaccine100"] = np.nan

    # type binaire pour Vaccine100 si possible
    if df["Vaccine100"].notna().any():
        df["Vaccine100"] = df["Vaccine100"].astype("float").round().astype("Int64")

    return df


def handle_missing_data(df: pd.DataFrame, strategy: str = "keep", threshold: float = 0.5) -> pd.DataFrame:
    """
    Gère les valeurs manquantes selon la stratégie choisie.
    
    Args:
        df: DataFrame à traiter
        strategy: "keep", "drop_rows", ou "drop_cols"
        threshold: seuil de % de valeurs manquantes pour drop_cols (0.5 = 50%)
    
    Returns:
        DataFrame traité
    """
    initial_shape = df.shape
    
    if strategy == "keep":
        print(f"📊 Stratégie 'keep': conservation de toutes les données ({initial_shape[0]} lignes, {initial_shape[1]} colonnes)")
        return df
    
    elif strategy == "drop_rows":
        df_clean = df.dropna()
        print(f"📊 Stratégie 'drop_rows': {initial_shape[0]} → {df_clean.shape[0]} lignes ({initial_shape[0] - df_clean.shape[0]} supprimées)")
        return df_clean
    
    elif strategy == "drop_cols":
        # Calculer le % de valeurs manquantes par colonne
        missing_pct = df.isnull().sum() / len(df)
        cols_to_keep = missing_pct[missing_pct <= threshold].index.tolist()
        cols_dropped = [col for col in df.columns if col not in cols_to_keep]
        
        df_clean = df[cols_to_keep]
        print(f"📊 Stratégie 'drop_cols' (seuil={threshold*100}%): {initial_shape[1]} → {df_clean.shape[1]} colonnes ({len(cols_dropped)} supprimées)")
        if cols_dropped:
            print(f"   Colonnes supprimées: {', '.join(cols_dropped[:10])}" + ("..." if len(cols_dropped) > 10 else ""))
        return df_clean
    
    else:
        raise ValueError(f"Stratégie inconnue: {strategy}. Options: 'keep', 'drop_rows', 'drop_cols'")


def load_and_combine_sheets(excel_path: Path, sheets: list[str], common_only: bool = True) -> pd.DataFrame:
    """
    Charge les feuilles listées, harmonise les colonnes clés, concatène.
    Si common_only=True, ne garde que les colonnes communes à toutes les feuilles.
    """
    xls = pd.ExcelFile(excel_path)
    
    all_dfs = []

    for sheet in sheets:
        if sheet not in xls.sheet_names:
            print(f"⚠️ Feuille {sheet} absente du fichier, ignorée.")
            continue

        df = pd.read_excel(excel_path, sheet_name=sheet)
        
        # D'ABORD standardiser les noms de colonnes
        df = standardize_columns(df, sheet)
        
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("Aucune feuille chargée, vérifier les noms de sheets.")

    # ENSUITE obtenir les colonnes communes (après standardisation)
    if common_only:
        sheets_columns = [set(df.columns) for df in all_dfs]
        common_columns = set.intersection(*sheets_columns)
        print(f"✅ {len(common_columns)} colonnes communes trouvées après standardisation")
        
        # Filtrer chaque df
        all_dfs = [df[sorted(common_columns)] for df in all_dfs]

    # Concaténation avec union de colonnes (pandas gère ça automatiquement)
    combined = pd.concat(all_dfs, axis=0, ignore_index=True)

    # Option : enlever les doublons EXACTS (toutes colonnes identiques)
    combined = combined.drop_duplicates()

    return combined


# -------------------------------------------------------------------
# 3. Utilisation
# -------------------------------------------------------------------
combined_df = load_and_combine_sheets(DATA_PATH, SHEETS_TO_LOAD, common_only=True)

print(f"\n{'='*60}")
print(f"Avant traitement des valeurs manquantes: {combined_df.shape}")
print(f"{'='*60}")

# Gestion des valeurs manquantes
combined_df = handle_missing_data(combined_df, strategy=MISSING_DATA_STRATEGY, threshold=MISSING_THRESHOLD)

print(f"\n{'='*60}")
print(f"✅ Forme finale: {combined_df.shape}")
print(f"{'='*60}")

# Affichage des statistiques de valeurs manquantes
missing_stats = combined_df.isnull().sum()
if missing_stats.sum() > 0:
    print(f"\nValeurs manquantes par colonne (top 10):")
    print(missing_stats[missing_stats > 0].sort_values(ascending=False).head(10))
else:
    print(f"\n✅ Aucune valeur manquante dans le dataset final")

print(f"\nAperçu des données:")
print(combined_df[["source_sheet", "OS_months", "OS_event", "Vaccine100"]].head())

combined_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"\n✅ Données exportées vers {OUTPUT_PATH}")