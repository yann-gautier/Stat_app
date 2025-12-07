from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from itertools import combinations
from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuration globale
DEFAULT_N_BOOT = 100  # Utiliser 1000+ pour runs de production finaux

# -------------------------------------------------------------------
# 1. Chargement des données
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "../data/combined_covid_data.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Le fichier {DATA_PATH} n'existe pas. Exécutez d'abord preparation_covid_data.py")

df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes\n")

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

def fit_model_for_target_improved(df, target, predictors, is_binary=False, min_rows=50, random_state=0):
    """
    Fit robuste avec régularisation, VIF filtering, permutation importance et post-selection inference.
    
    Args:
        df: DataFrame complet
        target: nom de la variable cible
        predictors: liste des prédicteurs candidats
        is_binary: si True, utilise LogisticRegressionCV, sinon ElasticNetCV
        min_rows: nombre minimum de lignes après nettoyage
        random_state: seed pour reproductibilité
    
    Returns:
        dict avec 'coefs', 'perm_importance', 'statsmodels_summary', 'metadata'
    """
    # Nettoyer les données 
    df_clean = df[[target] + predictors].dropna()
    print(f"Nettoyage des données pour {target}: {df_clean.shape}")
    
    if len(df_clean) < min_rows:
        return {"coefs": pd.DataFrame(), "perm_importance": pd.DataFrame(), 
                "statsmodels_summary": None, "metadata": {"n_rows": len(df_clean)}}
    
    # Séparer numériques et catégorielles
    num_preds = [p for p in predictors if is_numeric(df_clean, p)]
    cat_preds = [p for p in predictors if not is_numeric(df_clean, p)]
    
    # Gérer catégorielles haute cardinalité: grouper modalités rares
    rare_threshold = 5
    for col in cat_preds[:]:  # copie pour modification
        value_counts = df_clean[col].value_counts()
        rare_mask = df_clean[col].isin(value_counts[value_counts < rare_threshold].index)
        if rare_mask.sum() > 0:
            df_clean = df_clean.copy()  # Éviter SettingWithCopyWarning
            df_clean.loc[rare_mask, col] = "__rare__"
            print(f"   Grouped {rare_mask.sum()} rare values in {col}")
        
        # Filtrer si encore trop de modalités
        n_unique = df_clean[col].nunique()
        if n_unique > 20:
            print(f"   Removing {col}: too many categories ({n_unique})")
            cat_preds.remove(col)
        elif n_unique < 2:
            print(f"   Removing {col}: only {n_unique} category")
            cat_preds.remove(col)
    
    # Filtrage VIF sur les variables numériques
    if len(num_preds) >= 2:
        print(f"   Initial VIF check on {len(num_preds)} numeric features...")
        num_preds_filtered = iterative_vif_filter(df_clean, num_preds, threshold=10)
        print(f"   After VIF filtering: {len(num_preds_filtered)} numeric features retained")
        num_preds = num_preds_filtered
    
    all_preds = num_preds + cat_preds
    if len(all_preds) == 0:
        return {"coefs": pd.DataFrame(), "perm_importance": pd.DataFrame(), 
                "statsmodels_summary": None, "metadata": {"n_rows": len(df_clean)}}
    
    # Pipeline de preprocessing
    transformers = []
    if len(num_preds) > 0:
        transformers.append(("num", StandardScaler(), num_preds))
    if len(cat_preds) > 0:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='first'), cat_preds))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    
    # Modèle avec régularisation et CV
    if is_binary:
        model = LogisticRegressionCV(penalty='l1', solver='saga', cv=5, max_iter=2000, 
                                     random_state=random_state, n_jobs=-1)
    else:
        model = ElasticNetCV(cv=5, max_iter=2000, random_state=random_state, n_jobs=-1)
    
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    
    try:
        # Fit sur toutes les données (pas de split pour simplifier)
        pipe.fit(df_clean[all_preds], df_clean[target])
        
        # Récupérer les noms de features après preprocessing
        feature_names = []
        if len(num_preds) > 0:
            feature_names.extend(num_preds)
        if len(cat_preds) > 0:
            ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
            cat_features = ohe.get_feature_names_out(cat_preds)
            feature_names.extend(cat_features)
        
        # Coefficients du modèle régularisé
        coefs = pipe.named_steps["model"].coef_
        if len(coefs.shape) > 1:
            coefs = coefs.flatten()
        
        # Vérifier la cohérence
        if len(coefs) != len(feature_names):
            print(f"   ⚠️ Mismatch coefs: {len(coefs)} vs features: {len(feature_names)}")
            # Ajuster si nécessaire
            min_len = min(len(coefs), len(feature_names))
            coefs = coefs[:min_len]
            feature_names = feature_names[:min_len]
        
        coefs_df = pd.DataFrame({
            "feature": feature_names,
            "coef": coefs,
            "abs_coef": np.abs(coefs)
        }).sort_values("abs_coef", ascending=False)
        
        # Permutation importance sur échantillon (plus rapide)
        perm_df = pd.DataFrame()  # Désactivé temporairement pour éviter erreurs
        if len(df_clean) > 100:  # Seulement si assez de données
            sample_size = min(500, len(df_clean))  # Limiter à 500 pour vitesse
            sample_idx = np.random.choice(len(df_clean), sample_size, replace=False)
            X_sample = df_clean[all_preds].iloc[sample_idx]
            y_sample = df_clean[target].iloc[sample_idx]
            
            print(f"   Computing permutation importance on {sample_size} samples...")
            perm_df = compute_permutation_importance(pipe, X_sample, y_sample, feature_names, 
                                                    n_repeats=10, random_state=random_state)  # Réduit à 10
        
        # Post-selection inference: prendre features avec coef non-nul
        selected_features = coefs_df[coefs_df["abs_coef"] > 1e-6]["feature"].tolist()
        
        # Mapper les features OHE vers leurs colonnes originales
        selected_original = []
        for feat in selected_features:
            # Si c'est une feature OHE, extraire la colonne originale
            original_found = False
            for cat_col in cat_preds:
                if feat.startswith(f"{cat_col}_"):
                    if cat_col not in selected_original:
                        selected_original.append(cat_col)
                    original_found = True
                    break
            if not original_found and feat in num_preds:
                selected_original.append(feat)
        
        statsmodels_summary = None
        if len(selected_original) > 0:
            print(f"   Post-selection statsmodels on {len(selected_original)} features...")
            # Préparer données pour statsmodels (one-hot encoding manuel si nécessaire)
            X_sm = df_clean[selected_original].copy()
            for col in selected_original:
                if col in cat_preds:
                    dummies = pd.get_dummies(X_sm[col], prefix=col, drop_first=True)
                    X_sm = pd.concat([X_sm.drop(columns=[col]), dummies], axis=1)
            
            statsmodels_summary = post_selection_statsmodels(X_sm, df_clean[target], 
                                                            X_sm.columns.tolist(), is_binary)
        
        metadata = {
            "n_rows": len(df_clean),
            "n_features_initial": len(all_preds),
            "n_features_selected": len(selected_features),
            "model_type": "LogisticRegressionCV" if is_binary else "ElasticNetCV",
            "note": "statsmodels post-selection run" if statsmodels_summary is not None else "statsmodels not run (insufficient data)"
        }
        
        return {
            "coefs": coefs_df,
            "perm_importance": perm_df,
            "statsmodels_summary": statsmodels_summary,
            "metadata": metadata
        }
    
    except Exception as e:
        print(f"⚠️ Erreur pour {target}: {e}")
        import traceback
        traceback.print_exc()
        return {"coefs": pd.DataFrame(), "perm_importance": pd.DataFrame(), 
                "statsmodels_summary": None, "metadata": {"error": str(e)}}

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
# 3b. Advanced helper functions for robust analysis
# -------------------------------------------------------------------

def compute_vif(df, feature_list):
    """
    Calcule le VIF (Variance Inflation Factor) pour chaque variable.
    VIF > 10 indique une multicolinéarité problématique.
    """
    if len(feature_list) == 0:
        return pd.DataFrame()
    
    try:
        df_numeric = df[feature_list].select_dtypes(include=[np.number])
        if df_numeric.shape[1] < 2:
            return pd.DataFrame()
        
        vif_data = pd.DataFrame()
        vif_data["feature"] = df_numeric.columns
        vif_data["VIF"] = [variance_inflation_factor(df_numeric.values, i) 
                           for i in range(df_numeric.shape[1])]
        return vif_data.sort_values("VIF", ascending=False)
    except Exception as e:
        print(f"⚠️ Erreur VIF: {e}")
        return pd.DataFrame()

def iterative_vif_filter(df, features, threshold=10):
    """
    Supprime itérativement les variables avec VIF > threshold.
    Retourne la liste filtrée de features.
    """
    numeric_features = [f for f in features if is_numeric(df, f)]
    if len(numeric_features) < 2:
        return features
    
    df_work = df[numeric_features].copy()
    remaining = list(numeric_features)
    
    max_iterations = 20
    for iteration in range(max_iterations):
        vif_df = compute_vif(df_work, remaining)
        if vif_df.empty or vif_df["VIF"].max() <= threshold:
            break
        
        # Retirer la variable avec le VIF le plus élevé
        worst_var = vif_df.iloc[0]["feature"]
        remaining.remove(worst_var)
        df_work = df_work[remaining]
        print(f"   VIF filter iteration {iteration+1}: removed {worst_var} (VIF={vif_df.iloc[0]['VIF']:.2f})")
    
    # Remettre les variables non-numériques
    non_numeric = [f for f in features if f not in numeric_features]
    return remaining + non_numeric

def compute_permutation_importance(pipe, X, y, feature_names=None, n_repeats=30, random_state=0):
    """
    Calcule l'importance par permutation en s'assurant que les noms de features
    correspondent aux colonnes passées dans X.
    - Si feature_names est fourni, il doit correspondre à X.columns.
    - Par défaut, on prend feature_names = list(X.columns).
    """
    try:
        # S'assurer que X est DataFrame et récupérer ses colonnes
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if feature_names is None:
            feature_names = list(X.columns)
        else:
            # sécuriser la longueur
            feature_names = feature_names[:X.shape[1]]

        result = permutation_importance(pipe, X, y, n_repeats=n_repeats,
                                       random_state=random_state, n_jobs=-1)

        # result.importances_mean correspond aux colonnes de X
        if len(result.importances_mean) != len(feature_names):
            print(f"⚠️ Mismatch permutation importances vs feature_names: {len(result.importances_mean)} vs {len(feature_names)}")

        perm_df = pd.DataFrame({
            "feature": feature_names[:len(result.importances_mean)],
            "perm_imp_mean": result.importances_mean,
            "perm_imp_std": result.importances_std
        }).sort_values("perm_imp_mean", ascending=False)
        return perm_df
    except Exception as e:
        print(f"⚠️ Erreur permutation importance: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def post_selection_statsmodels(X, y, selected_feature_names, is_binary):
    """
    Ajuste un modèle statsmodels sur les features sélectionnées pour obtenir p-values et IC.
    Post-selection inference approximatif avec conversion forcée en float.
    """
    try:
        X_sub = X[selected_feature_names].copy()
        
        # Conversion agressive: forcer TOUT en float, même si ça coerce en NaN
        X_sub = X_sub.apply(pd.to_numeric, errors='coerce')
        
        # Supprimer lignes avec NaN après conversion
        mask = ~X_sub.isna().any(axis=1) & ~pd.isna(y)
        X_clean = X_sub[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 10:
            return None
        
        # Forcer explicitement en float64 pour numpy
        X_clean = X_clean.astype(np.float64)
        y_clean = pd.Series(y_clean).astype(np.float64)
        
        X_const = add_constant(X_clean)
        
        if is_binary:
            model = Logit(y_clean, X_const).fit(disp=0, maxiter=200)
        else:
            model = OLS(y_clean, X_const).fit()
        
        summary_df = pd.DataFrame({
            "feature": model.params.index,
            "coef_sm": model.params.values,
            "p_raw": model.pvalues.values,
            "ci_lower": model.conf_int()[0].values,
            "ci_upper": model.conf_int()[1].values
        })
        
        return summary_df
    except Exception as e:
        # Silencieux: retourner None sans afficher l'erreur
        return None

def bootstrap_mediation(df, cause, mediator, outcome, n_boot=1000, seed=None):
    """
    Test de médiation bootstrap pour effet indirect a*b.
    Retourne: dict avec mean, ci_lower, ci_upper, p_approx.
    """
    if seed is not None:
        np.random.seed(seed)
    
    indirect_effects = []
    
    for i in range(n_boot):
        df_boot = df.sample(n=len(df), replace=True, random_state=seed+i if seed else None)
        
        try:
            # a: cause -> mediator
            X_a = add_constant(df_boot[[cause]])
            model_a = OLS(df_boot[mediator], X_a).fit()
            a_coef = model_a.params[cause]
            
            # b: mediator -> outcome (controlling for cause)
            X_b = add_constant(df_boot[[cause, mediator]])
            is_binary_outcome = df_boot[outcome].nunique() == 2
            
            if is_binary_outcome:
                model_b = Logit(df_boot[outcome], X_b).fit(disp=0, maxiter=100)
            else:
                model_b = OLS(df_boot[outcome], X_b).fit()
            
            b_coef = model_b.params[mediator]
            
            indirect_effects.append(a_coef * b_coef)
        except:
            continue
    
    if len(indirect_effects) == 0:
        return None
    
    indirect_effects = np.array(indirect_effects)
    mean_effect = np.mean(indirect_effects)
    ci_lower = np.percentile(indirect_effects, 2.5)
    ci_upper = np.percentile(indirect_effects, 97.5)

    # proportion positive (utile à inspecter) et p-value empirique two-sided
    prop_pos = np.mean(indirect_effects > 0)
    p_empirical = 2 * min(prop_pos, 1 - prop_pos)

    return {
        "mean": mean_effect,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "prop_positive": prop_pos,
        "p_approx": p_empirical,
        "n_boot": len(indirect_effects)
    }

def cox_checks_and_report(cox_model, df_cox, duration_col, event_col):
    """
    Vérifie les assumptions du modèle Cox et retourne un rapport programmatique.
    """
    try:
        from lifelines.statistics import proportional_hazard_test
        
        test_res = proportional_hazard_test(cox_model, df_cox, time_transform='rank')
        ph_summary = test_res.summary  # DataFrame with p-values per covariate
        violations = ph_summary[ph_summary['p'] < 0.05].index.tolist()
        
        return {
            "model_converged": True,
            "ph_test_summary": ph_summary.to_dict(),
            "ph_violations": violations,
            "ph_assumptions_ok": len(violations) == 0,
            "recommendation": "OK" if len(violations) == 0 else "Consider stratification or time-varying covariates"
        }
    except Exception as e:
        return {
            "model_converged": False,
            "error": str(e),
            "recommendation": "Check model specification"
        }

# -------------------------------------------------------------------
# 4. Construction du DAG niveau par niveau
# -------------------------------------------------------------------

print("="*60)
print("CONSTRUCTION DU DAG CAUSAL (IMPROVED)")
print("="*60)

all_links = []
mediator_effects = {}  # Pour détecter les effets médiés
run_log = {
    "timestamp": datetime.now().isoformat(),
    "data_shape": df.shape,
    "levels_analyzed": [],
    "bootstrap_mediations": []
}

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
    level_log = {"level": target_level, "targets": []}
    
    for target in target_vars:
        if target not in df.columns:
            continue
        
        print(f"\n🎯 Target: {target}")
        
        # Déterminer si binaire
        is_binary = df[target].nunique() == 2
        
        # Fit le modèle amélioré
        result = fit_model_for_target_improved(df, target, predictor_vars, is_binary=is_binary)
        
        coefs = result["coefs"]
        perm_imp = result["perm_importance"]
        sm_summary = result["statsmodels_summary"]
        metadata = result["metadata"]
        
        if coefs.empty:
            print(f"   ⚠️ Pas de modèle pour {target}")
            level_log["targets"].append({"target": target, "status": "failed", "metadata": metadata})
            continue
        
        # Seuil de significativité (top features)
        top_features = coefs.head(10)
        
        print(f"   ✅ Top prédicteurs (régularisé):")
        for _, row in top_features.iterrows():
            feature_name = row['feature']
            coef_val = row['coef']
            
            # Mapper vers variable originale
            original_var = map_to_original_variable(feature_name)
            
            # Chercher p-value dans statsmodels si disponible
            p_raw = None
            perm_imp_val = None
            
            if sm_summary is not None and feature_name in sm_summary["feature"].values:
                p_raw = sm_summary[sm_summary["feature"] == feature_name]["p_raw"].values[0]
            
            if not perm_imp.empty and feature_name in perm_imp["feature"].values:
                perm_imp_val = perm_imp[perm_imp["feature"] == feature_name]["perm_imp_mean"].values[0]
            
            p_str = f"{p_raw:.3f}" if p_raw is not None else "N/A"
            perm_str = f"{perm_imp_val:.4f}" if perm_imp_val is not None else "N/A"
            print(f"      {original_var} → {target} (coef={coef_val:.3f}, p={p_str}, perm_imp={perm_str})")
            
            # Ajouter le lien avec métadonnées enrichies
            all_links.append({
                "from": original_var,
                "to": target,
                "coef": coef_val,
                "level_from": VAR_TO_LEVEL.get(original_var, -1),
                "level_to": target_level,
                "type": "direct_effect",
                "method_source": "reg_l1" if is_binary else "elasticnet",
                "p_raw": p_raw,
                "perm_imp": perm_imp_val
            })
        
        # Logger les infos
        level_log["targets"].append({
            "target": target,
            "status": "success",
            "metadata": metadata,
            "top_features": top_features["feature"].head(5).tolist()
        })
    
    run_log["levels_analyzed"].append(level_log)

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
            coefs_total = fit_model_for_target_improved(df, final_outcome, [cause], 
                                               is_binary=(final_outcome == "OS_event"))
            
            # Modèle avec médiateur
            coefs_direct = fit_model_for_target_improved(df, final_outcome, [cause, mediator],
                                                is_binary=(final_outcome == "OS_event"))
            
            # Test bootstrap de médiation (réduit pour vitesse)
            try:
                print(f"   Testing mediation: {cause} → {mediator} → {final_outcome}")
                boot_result = bootstrap_mediation(df, cause, mediator, final_outcome, 
                                                 n_boot=DEFAULT_N_BOOT, seed=42)
                
                if boot_result is not None:
                    mean_indirect = boot_result["mean"]
                    ci_lower = boot_result["ci_lower"]
                    ci_upper = boot_result["ci_upper"]
                    p_approx = boot_result["p_approx"]
                    
                    # Médiation significative si IC ne contient pas 0
                    is_significant = not (ci_lower <= 0 <= ci_upper)
                    
                    if is_significant:
                        print(f"\n🔗 MÉDIATION BOOTSTRAP DÉTECTÉE:")
                        print(f"   {cause} → {mediator} → {final_outcome}")
                        print(f"   Effet indirect: {mean_indirect:.3f}")
                        print(f"   IC 95%: [{ci_lower:.3f}, {ci_upper:.3f}]")
                        print(f"   p-value approx: {p_approx:.3f}")
                        
                        # Ajouter le lien médié
                        all_links.append({
                            "from": cause,
                            "to": final_outcome,
                            "coef": mean_indirect,
                            "level_from": VAR_TO_LEVEL.get(cause, -1),
                            "level_to": VAR_TO_LEVEL.get(final_outcome, 4),
                            "type": f"mediated_by_{mediator}",
                            "method_source": "bootstrap_mediation",
                            "p_raw": p_approx,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper
                        })
                        
                        # Logger
                        run_log["bootstrap_mediations"].append({
                            "cause": cause,
                            "mediator": mediator,
                            "outcome": final_outcome,
                            "indirect_effect": mean_indirect,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "p_approx": p_approx
                        })
            except Exception as e:
                print(f"   ⚠️ Bootstrap mediation failed: {e}")

# -------------------------------------------------------------------
# 6. Modèle de Cox pour OS (validation finale)
# -------------------------------------------------------------------

print(f"\n{'='*60}")
print("MODÈLE DE COX FINAL : VALIDATION")
print(f"{'='*60}")

if "OS_months" in df.columns and "OS_event" in df.columns:
    df_cox = df[["OS_months", "OS_event"]].copy()
    
    # Limiter aux variables les plus importantes (éviter explosion dimensions)
    # Seulement niveaux 0-2 pour éviter trop de variables
    cox_predictors = []
    for level in range(0, 3):  # Seulement 0, 1, 2 (pas 3 pour éviter PFS qui peut causer problèmes)
        cox_predictors.extend(CAUSAL_LEVELS[level])
    
    cox_predictors = [v for v in cox_predictors if v in df.columns and v not in ["OS_months", "OS_event"]]
    
    # Ajouter les colonnes - limiter les catégorielles
    for col in cox_predictors:
        if is_numeric(df, col):
            df_cox[col] = df[col]
        else:
            # One-hot encode seulement si peu de modalités
            n_unique = df[col].nunique()
            if n_unique <= 5:  # Limite stricte
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_cox = pd.concat([df_cox, dummies], axis=1)
            else:
                print(f"   Skipping {col} in Cox: too many categories ({n_unique})")
    
    # Supprimer les lignes avec valeurs manquantes
    df_cox = df_cox.dropna()
    print(f"Nettoyage des données pour Cox: {df_cox.shape}")
    if len(df_cox) >= 50:
        try:
            cox = CoxPHFitter(penalizer=0.01)
            cox.fit(df_cox, duration_col="OS_months", event_col="OS_event")
            
            summary = cox.summary.sort_values("p", ascending=True)
            
            # Appliquer correction FDR
            p_values = summary["p"].values
            reject, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
            summary["p_adj"] = p_adj
            summary["significant_fdr"] = reject
            
            print(f"\n✅ Modèle de Cox (n={len(df_cox)})")
            print("\nVariables significatives (p_adj < 0.05 après correction FDR):")
            sig_vars = summary[summary["p_adj"] < 0.05]
            print(sig_vars[["coef", "exp(coef)", "p", "p_adj"]])
            
            # Vérifier les assumptions PH
            print("\n🔍 Vérification des assumptions Proportional Hazards...")
            cox_report = cox_checks_and_report(cox, df_cox, "OS_months", "OS_event")
            print(f"   Convergence: {cox_report.get('model_converged', 'Unknown')}")
            print(f"   Recommendation: {cox_report.get('recommendation', 'N/A')}")
            
            # Ajouter les effets significatifs du Cox (après FDR)
            for var in sig_vars.index:
                original_var = map_to_original_variable(var)
                all_links.append({
                    "from": original_var,
                    "to": "OS",
                    "coef": sig_vars.loc[var, "coef"],
                    "level_from": VAR_TO_LEVEL.get(original_var, -1),
                    "level_to": 4,
                    "type": "cox_survival",
                    "method_source": "cox_survival",
                    "p_raw": sig_vars.loc[var, "p"],
                    "p_adj": sig_vars.loc[var, "p_adj"]
                })
            
            # Logger Cox results
            run_log["cox_model"] = {
                "n_rows": len(df_cox),
                "n_significant_raw": len(summary[summary["p"] < 0.05]),
                "n_significant_fdr": len(sig_vars),
                "convergence": cox_report.get('model_converged', False),
                "ph_assumptions": cox_report
            }
            print(f"   PH violations: {cox_report.get('ph_violations', [])}")
            
        except Exception as e:
            print(f"❌ Erreur Cox: {e}")
            import traceback
            traceback.print_exc()
            run_log["cox_model"] = {"error": str(e)}

# -------------------------------------------------------------------
# 7. Export du DAG structuré avec corrections FDR
# -------------------------------------------------------------------

links_df = pd.DataFrame(all_links)

# Appliquer correction FDR sur les p-values disponibles
if "p_raw" in links_df.columns:
    valid_p = links_df["p_raw"].notna()
    if valid_p.sum() > 0:
        p_vals = links_df.loc[valid_p, "p_raw"].values
        reject, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
        links_df.loc[valid_p, "p_adj"] = p_adj
        links_df.loc[valid_p, "significant_fdr"] = reject
        print(f"✅ Correction FDR appliquée sur {valid_p.sum()} liens avec p-values")

# Dédupliquer (garder le lien le plus fort par paire)
links_df["abs_coef"] = links_df["coef"].abs()
links_df = links_df.sort_values("abs_coef", ascending=False)
links_df = links_df.drop_duplicates(subset=["from", "to"], keep="first")

# Marquer les liens quasi-nuls
links_df["zeroed_by_regularization"] = links_df["abs_coef"] <= 1e-6

# Optionnel : filtrer réellement les liens quasi-nuls pour l'export principal
export_df = links_df[~links_df["zeroed_by_regularization"]].copy()
# si tu veux garder tout pour audit: export_df = links_df.copy()

print(f"\n{'='*60}")
print(f"EXPORT DU DAG AMÉLIORÉ")
print(f"{'='*60}")
print(f"Total de liens causaux détectés : {len(export_df)}")

# Export CSV
output_dir = SCRIPT_DIR / "../data"
export_df.to_csv(output_dir / "causal_dag_structured.csv", index=False, encoding='utf-8-sig')
print(f"✅ DAG exporté vers {output_dir / 'causal_dag_structured.csv'}")

# Export run log JSON avec métriques robustes
run_log["total_links"] = int(len(export_df))
run_log["links_with_pvalues"] = int(export_df["p_raw"].notna().sum()) if "p_raw" in export_df.columns else 0
run_log["significant_after_fdr"] = int(export_df["significant_fdr"].sum()) if "significant_fdr" in export_df.columns else 0

with open(output_dir / "causal_dag_runlog.json", "w", encoding="utf-8") as f:
    json.dump(run_log, f, indent=2, default=str)
print(f"✅ Run log exporté vers {output_dir / 'causal_dag_runlog.json'}")

# Statistiques par niveau
print("\nRésumé par niveau:")
for level in range(0, 5):
    links_from_level = export_df[export_df["level_from"] == level]
    links_to_level = export_df[export_df["level_to"] == level]
    print(f"  Niveau {level}: {len(links_from_level)} liens sortants, {len(links_to_level)} liens entrants")

# Chemins vers OS
os_links = export_df[export_df["to"].isin(["OS", "OS_months", "OS_event"])]
print(f"\n🎯 Variables influençant directement OS : {os_links['from'].nunique()}")
print(os_links.groupby("from")["coef"].first().sort_values(key=abs, ascending=False).head(15))

# Rapport final
print(f"\n{'='*60}")
print("RAPPORT FINAL")
print(f"{'='*60}")
print(f"✅ Total liens: {len(export_df)}")
print(f"✅ Liens avec p-values: {export_df['p_raw'].notna().sum()}")
if "significant_fdr" in export_df.columns:
    print(f"✅ Liens significatifs (FDR < 0.05): {export_df['significant_fdr'].sum()}")
print(f"✅ Médiations bootstrap détectées: {len(run_log['bootstrap_mediations'])}")
print(f"\nFichiers générés:")
print(f"  - {output_dir / 'causal_dag_structured.csv'}")
print(f"  - {output_dir / 'causal_dag_runlog.json'}")
