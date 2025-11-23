from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# 1. Chargement des données
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "../ressources/combined_covid_data.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Le fichier {DATA_PATH} n'existe pas. Exécutez d'abord preparation_covid_data.py")

df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes\n")

# -------------------------------------------------------------------
# 2. Définition de la structure causale hiérarchique
# -------------------------------------------------------------------

# Niveaux de causalité (du plus amont au plus aval)
CAUSAL_LEVELS = {
    # Niveau 0 : Variables démographiques (non modifiables, toujours en amont)
    0: ["Gender", "Ethnicity", "Age_at_ICI_start"],
    
    # Niveau 1 : Caractéristiques baseline de la maladie
    1: ["Simplified_Stage", "ECOG", "CNS_disease", "Previous_history_of_malignancy_at_ICI_start"],
    
    # Niveau 2 : Traitements et interventions
    2: ["Vaccine100", "Steroid_win_1_month_of_Vaccine", "Concurrent_Chemo", "BRAF"],
    
    # Niveau 3 : Outcomes intermédiaires (médiateurs)
    3: ["PFS_", "PFS_Code"],
    
    # Niveau 4 : Outcome final (puits)
    4: ["OS_months", "OS_event"]
}

# Colonnes à exclure totalement
EXCLUDE_COLS = ["source_sheet", "Immunotherapy_Agent", "PathologicDx"]

# Inverse mapping : variable → niveau
VAR_TO_LEVEL = {}
for level, vars in CAUSAL_LEVELS.items():
    for var in vars:
        VAR_TO_LEVEL[var] = level

print("="*60)
print("STRUCTURE CAUSALE HIÉRARCHIQUE")
print("="*60)
for level, vars in CAUSAL_LEVELS.items():
    print(f"Niveau {level}: {vars}")
print()

# -------------------------------------------------------------------
# 3. Helper functions
# -------------------------------------------------------------------

def is_numeric(df, col):
    """Vérifie si une colonne est numérique"""
    return pd.api.types.is_numeric_dtype(df[col])

def fit_model_for_target(df, target, predictors, is_binary=False):
    """
    Fit un modèle pour prédire target à partir de predictors.
    Retourne les coefficients/importances.
    """
    # Nettoyer les données
    df_clean = df[[target] + predictors].dropna()
    
    if len(df_clean) < 50:
        return pd.DataFrame()
    
    # Séparer numériques et catégorielles
    num_preds = [p for p in predictors if is_numeric(df_clean, p)]
    cat_preds = [p for p in predictors if not is_numeric(df_clean, p)]
    
    # Filtrer les catégorielles avec trop de modalités
    cat_preds = [p for p in cat_preds if df_clean[p].nunique() <= 20]
    
    if len(num_preds) + len(cat_preds) == 0:
        return pd.DataFrame()
    
    # Pipeline de preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_preds),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='first'), cat_preds),
        ],
        remainder="drop"
    )
    
    # Modèle selon le type de target
    if is_binary:
        model = LogisticRegression(max_iter=500, penalty="l2", C=1.0)
    else:
        model = LinearRegression()
    
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    
    try:
        pipe.fit(df_clean[predictors], df_clean[target])
        
        # Récupérer les noms de features
        ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
        cat_features = ohe.get_feature_names_out(cat_preds) if len(cat_preds) > 0 else []
        all_features = np.concatenate([num_preds, cat_features])
        
        # Coefficients
        coefs = pipe.named_steps["model"].coef_
        if len(coefs.shape) > 1:
            coefs = coefs.flatten()
        
        result = pd.DataFrame({
            "feature": all_features,
            "coef": coefs,
            "abs_coef": np.abs(coefs)
        }).sort_values("abs_coef", ascending=False)
        
        return result
    
    except Exception as e:
        print(f"⚠️ Erreur pour {target}: {e}")
        return pd.DataFrame()

def map_to_original_variable(encoded_name):
    """
    Mappe une variable encodée vers sa variable originale.
    Exemple: Gender_Male → Gender
    """
    for base_var in VAR_TO_LEVEL.keys():
        if encoded_name.startswith(f"{base_var}_"):
            return base_var
    return encoded_name

# -------------------------------------------------------------------
# 4. Construction du DAG niveau par niveau
# -------------------------------------------------------------------

print("="*60)
print("CONSTRUCTION DU DAG CAUSAL")
print("="*60)

all_links = []
mediator_effects = {}  # Pour détecter les effets médiés

# Pour chaque niveau, modéliser toutes les variables en fonction des niveaux précédents
for target_level in range(1, 5):  # Niveaux 1 à 4
    target_vars = CAUSAL_LEVELS[target_level]
    
    # Prédicteurs = tous les niveaux strictement antérieurs
    predictor_vars = []
    for pred_level in range(0, target_level):
        predictor_vars.extend(CAUSAL_LEVELS[pred_level])
    
    # Filtrer les variables qui existent dans les données
    predictor_vars = [v for v in predictor_vars if v in df.columns]
    
    if len(predictor_vars) == 0:
        continue
    
    print(f"\n{'='*60}")
    print(f"NIVEAU {target_level}: {target_vars}")
    print(f"Prédicteurs candidats (niveaux 0-{target_level-1}): {predictor_vars}")
    print(f"{'='*60}")
    
    # Pour chaque variable cible du niveau
    for target in target_vars:
        if target not in df.columns:
            continue
        
        print(f"\n🎯 Target: {target}")
        
        # Déterminer si binaire
        is_binary = df[target].nunique() == 2
        
        # Fit le modèle
        coefs = fit_model_for_target(df, target, predictor_vars, is_binary=is_binary)
        
        if coefs.empty:
            print(f"   ⚠️ Pas de modèle pour {target}")
            continue
        
        # Seuil de significativité (top features)
        top_features = coefs.head(10)
        
        print(f"   ✅ Top prédicteurs:")
        for _, row in top_features.iterrows():
            feature_name = row['feature']
            coef_val = row['coef']
            
            # Mapper vers variable originale
            original_var = map_to_original_variable(feature_name)
            
            print(f"      {original_var} → {target} (coef={coef_val:.3f})")
            
            # Ajouter le lien
            all_links.append({
                "from": original_var,
                "to": target,
                "coef": coef_val,
                "level_from": VAR_TO_LEVEL.get(original_var, -1),
                "level_to": target_level,
                "type": "direct_effect"
            })

# -------------------------------------------------------------------
# 5. Analyse des effets médiés (test de médiation)
# -------------------------------------------------------------------

print(f"\n{'='*60}")
print("ANALYSE DES EFFETS MÉDIÉS")
print(f"{'='*60}")

# Identifier les chemins X → M → Y où:
# - X est un prédicteur de M
# - M est un prédicteur de Y
# - L'effet de X sur Y diminue quand on contrôle pour M

links_df = pd.DataFrame(all_links)

# Pour chaque outcome final (OS)
for final_outcome in ["OS_months", "OS_event"]:
    if final_outcome not in df.columns:
        continue
    
    # Trouver les prédicteurs directs de l'outcome
    direct_predictors = links_df[links_df["to"] == final_outcome]["from"].unique()
    
    # Pour chaque prédicteur direct
    for mediator in direct_predictors:
        if mediator not in df.columns:
            continue
        
        # Trouver les causes du médiateur
        causes_of_mediator = links_df[links_df["to"] == mediator]["from"].unique()
        
        for cause in causes_of_mediator:
            if cause not in df.columns:
                continue
            
            # Test: Effet total (cause → outcome) vs Effet direct (cause → outcome | mediator)
            
            # Modèle sans médiateur
            coefs_total = fit_model_for_target(df, final_outcome, [cause], 
                                               is_binary=(final_outcome == "OS_event"))
            
            # Modèle avec médiateur
            coefs_direct = fit_model_for_target(df, final_outcome, [cause, mediator],
                                                is_binary=(final_outcome == "OS_event"))
            
            if not coefs_total.empty and not coefs_direct.empty:
                # Récupérer les coefficients
                try:
                    total_effect = coefs_total[coefs_total["feature"] == cause]["coef"].values[0]
                    direct_effect = coefs_direct[coefs_direct["feature"] == cause]["coef"].values[0]
                    
                    # Si l'effet diminue significativement, c'est une médiation
                    mediation_ratio = abs(direct_effect) / (abs(total_effect) + 1e-6)
                    
                    if mediation_ratio < 0.7:  # Effet réduit de >30%
                        print(f"\n🔗 MÉDIATION DÉTECTÉE:")
                        print(f"   {cause} → {mediator} → {final_outcome}")
                        print(f"   Effet total: {total_effect:.3f}")
                        print(f"   Effet direct: {direct_effect:.3f}")
                        print(f"   Médiation: {(1-mediation_ratio)*100:.1f}%")
                        
                        # Ajouter le lien médié
                        all_links.append({
                            "from": cause,
                            "to": final_outcome,
                            "coef": total_effect - direct_effect,
                            "level_from": VAR_TO_LEVEL.get(cause, -1),
                            "level_to": VAR_TO_LEVEL.get(final_outcome, 4),
                            "type": f"mediated_by_{mediator}"
                        })
                except:
                    pass

# -------------------------------------------------------------------
# 6. Modèle de Cox pour OS (validation finale)
# -------------------------------------------------------------------

print(f"\n{'='*60}")
print("MODÈLE DE COX FINAL : VALIDATION")
print(f"{'='*60}")

if "OS_months" in df.columns and "OS_event" in df.columns:
    df_cox = df[["OS_months", "OS_event"]].copy()
    
    # Ajouter toutes les variables des niveaux 0-3 (exclure niveau 4 = outcomes)
    cox_predictors = []
    for level in range(0, 4):
        cox_predictors.extend(CAUSAL_LEVELS[level])
    
    cox_predictors = [v for v in cox_predictors if v in df.columns and v not in ["OS_months", "OS_event"]]
    
    # Ajouter les colonnes
    for col in cox_predictors:
        if is_numeric(df, col):
            df_cox[col] = df[col]
        else:
            # One-hot encode
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df_cox = pd.concat([df_cox, dummies], axis=1)
    
    # Supprimer les lignes avec valeurs manquantes
    df_cox = df_cox.dropna()
    
    if len(df_cox) >= 50:
        try:
            cox = CoxPHFitter(penalizer=0.01)
            cox.fit(df_cox, duration_col="OS_months", event_col="OS_event")
            
            summary = cox.summary.sort_values("p", ascending=True)
            
            print(f"\n✅ Modèle de Cox (n={len(df_cox)})")
            print("\nVariables significatives (p < 0.05):")
            sig_vars = summary[summary["p"] < 0.05]
            print(sig_vars[["coef", "exp(coef)", "p"]])
            
            # Ajouter les effets significatifs du Cox
            for var in sig_vars.index:
                original_var = map_to_original_variable(var)
                all_links.append({
                    "from": original_var,
                    "to": "OS",
                    "coef": sig_vars.loc[var, "coef"],
                    "level_from": VAR_TO_LEVEL.get(original_var, -1),
                    "level_to": 4,
                    "type": "cox_survival"
                })
        except Exception as e:
            print(f"❌ Erreur Cox: {e}")

# -------------------------------------------------------------------
# 7. Export du DAG structuré
# -------------------------------------------------------------------

links_df = pd.DataFrame(all_links)

# Dédupliquer (garder le lien le plus fort par paire)
links_df["abs_coef"] = links_df["coef"].abs()
links_df = links_df.sort_values("abs_coef", ascending=False)
links_df = links_df.drop_duplicates(subset=["from", "to"], keep="first")

print(f"\n{'='*60}")
print(f"EXPORT DU DAG")
print(f"{'='*60}")
print(f"Total de liens causaux détectés : {len(links_df)}")

links_df.to_csv("causal_dag_structured.csv", index=False, encoding='utf-8-sig')
print(f"✅ DAG exporté vers causal_dag_structured.csv")

# Statistiques par niveau
print("\nRésumé par niveau:")
for level in range(0, 5):
    links_from_level = links_df[links_df["level_from"] == level]
    links_to_level = links_df[links_df["level_to"] == level]
    print(f"  Niveau {level}: {len(links_from_level)} liens sortants, {len(links_to_level)} liens entrants")

# Chemins vers OS
os_links = links_df[links_df["to"].isin(["OS", "OS_months", "OS_event"])]
print(f"\n🎯 Variables influençant directement OS : {os_links['from'].nunique()}")
print(os_links.groupby("from")["coef"].first().sort_values(key=abs, ascending=False).head(15))
