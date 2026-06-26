"""
Recalibrator - Apprentissage et recalibrage automatique
=========================================================

Ce module permet au système de s'améliorer automatiquement après chaque match.
Il utilise les résultats réels pour ajuster les coefficients tactiques via
une régression logistique (descente de gradient ou optimisation scipy).

Fonctionnement :
1. Récupère l'historique des matchs dont le résultat est connu
2. Construit les features : différence de xG ajustée par les bonus tactiques
3. Optimise les coefficients pour maximiser la vraisemblance des résultats observés
4. Met à jour les coefficients dans la base de données

Le modèle prédictif est :
    P(victoire domicile) = 1 / (1 + exp(-(diff_xG_ajustee + intercept)))

où diff_xG_ajustee = xG_home - xG_away + bonus_tactique(strat_A, strat_B)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # Fonction sigmoïde (logistique)
import json
import sqlite3
from datetime import datetime
import warnings
import traceback

# Imports locaux
from db import (
    get_coefficients, update_coefficients, get_matchs_avec_resultat,
    get_nb_matchs_historique, set_config, DB_NAME
)
from matrix_builder import calculer_bonus_tactique


# ============================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================

def load_training_data():
    """
    Charge les matchs historiques avec résultat connu pour l'entraînement.
    
    Returns:
        tuple: (X, y) où :
            - X : DataFrame avec les features (xg_home, xg_away, strategies, etc.)
            - y : Series avec le résultat (1 = victoire domicile, 0 = sinon)
    """
    df = get_matchs_avec_resultat()
    
    if df.empty:
        print("⚠️  Aucun match avec résultat trouvé dans l'historique.")
        return None, None
    
    print(f"📊 {len(df)} matchs chargés pour le recalibrage.")
    
    # Créer la variable cible : 1 si victoire domicile, 0 sinon
    y = (df['resultat'] == 'home').astype(int)
    
    # Distribution des résultats
    nb_home = (df['resultat'] == 'home').sum()
    nb_away = (df['resultat'] == 'away').sum()
    nb_draw = (df['resultat'] == 'draw').sum()
    print(f"   Victoires domicile : {nb_home} ({nb_home/len(df)*100:.1f}%)")
    print(f"   Victoires extérieur : {nb_away} ({nb_away/len(df)*100:.1f}%)")
    print(f"   Matchs nuls : {nb_draw} ({nb_draw/len(df)*100:.1f}%)")
    
    return df, y


def extract_strategies(df):
    """
    Extrait les stratégies depuis la colonne JSON.
    Si les stratégies ne sont pas disponibles, utilise des valeurs par défaut.
    
    Args:
        df: DataFrame avec colonne 'strategies' (JSON)
    
    Returns:
        DataFrame avec colonnes 'strat_A' et 'strat_B' ajoutées
    """
    df = df.copy()
    
    def safe_extract(row):
        try:
            if pd.notna(row['strategies']) and row['strategies']:
                strategies = json.loads(row['strategies'])
                return pd.Series({
                    'strat_A': strategies.get('A', 'ailes'),
                    'strat_B': strategies.get('B', 'pressing_haut')
                })
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        
        # Fallback : stratégies par défaut
        return pd.Series({
            'strat_A': 'ailes',
            'strat_B': 'pressing_haut'
        })
    
    strategies_df = df.apply(safe_extract, axis=1)
    df['strat_A'] = strategies_df['strat_A']
    df['strat_B'] = strategies_df['strat_B']
    
    return df


def build_features(df, coefficients):
    """
    Construit la matrice de features pour le modèle logistique.
    
    La feature principale est la différence de xG ajustée :
        diff_xg_adj = xg_home - xg_away + bonus_tactique(strat_A, strat_B)
    
    Args:
        df: DataFrame avec xg_home, xg_away, strat_A, strat_B
        coefficients: dict des coefficients tactiques actuels
    
    Returns:
        numpy.array: features (diff_xg_adj pour chaque match)
    """
    features = []
    
    for _, row in df.iterrows():
        xg_home = row.get('xg_home', 1.5)
        xg_away = row.get('xg_away', 1.3)
        strat_A = row.get('strat_A', 'ailes')
        strat_B = row.get('strat_B', 'pressing_haut')
        
        # Bonus tactique selon les stratégies
        bonus = calculer_bonus_tactique(strat_A, strat_B, coefficients)
        
        # Différence de xG ajustée
        diff_xg_adj = xg_home - xg_away + bonus
        
        features.append(diff_xg_adj)
    
    return np.array(features)


# ============================================================
# 2. MODÈLE LOGISTIQUE
# ============================================================

def sigmoid(z):
    """
    Fonction sigmoïde (logistique).
    Convertit un score en probabilité entre 0 et 1.
    """
    # Version stable numériquement
    return expit(z)


def predict_proba(features, params):
    """
    Prédit la probabilité de victoire domicile.
    
    Args:
        features: array de différences de xG ajustées
        params: dict avec les coefficients et l'intercept
    
    Returns:
        array de probabilités
    """
    z = features + params.get('intercept', 0.0)
    return sigmoid(z)


def log_likelihood_simple(params, features, y):
    """
    Version simplifiée pour le recalibrage avec un seul paramètre (intercept).
    Utile quand on n'a pas les données de stratégies.
    
    Args:
        params: [intercept]
        features: array 1D de différences de xG
        y: cible
    
    Returns:
        float: négatif de la log-vraisemblance
    """
    intercept = params[0]
    z = features + intercept
    probs = sigmoid(z)
    
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)
    
    ll = y * np.log(probs) + (1 - y) * np.log(1 - probs)
    return -np.sum(ll)


# ============================================================
# 3. RECALIBRAGE PRINCIPAL
# ============================================================

def recalibrate(method='auto', verbose=True):
    """
    Fonction principale de recalibrage automatique.
    
    Étapes :
    1. Charge l'historique des matchs
    2. Prépare les features et la cible
    3. Optimise les coefficients par maximum de vraisemblance
    4. Sauvegarde les nouveaux coefficients
    
    Args:
        method: 'auto' (choisit la meilleure), 'full' (tous les coeffs),
                'simple' (intercept seul)
        verbose: Si True, affiche les détails
    
    Returns:
        dict: Résumé du recalibrage
    """
    
    if verbose:
        print("\n" + "=" * 60)
        print("🔄 RECALIBRAGE AUTOMATIQUE DU MODÈLE")
        print("=" * 60)
        print(f"   Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # 1. Chargement des données
    df, y = load_training_data()
    
    if df is None or len(df) < 5:
        if verbose:
            print("⚠️  Pas assez de données pour recalibrer (minimum 5 matchs).")
            print("   Le modèle conserve ses coefficients actuels.")
        return {
            'success': False,
            'raison': 'Pas assez de données',
            'nb_matchs': len(df) if df is not None else 0
        }
    
    # 2. Extraction des stratégies
    df = extract_strategies(df)
    
    # 3. Récupération des coefficients actuels
    current_coeffs = get_coefficients()
    
    if verbose:
        print(f"\n📊 Coefficients actuels :")
        for k, v in current_coeffs.items():
            print(f"   {k}: {v:.4f}")
    
    # 4. Construction des features
    # Mode simple : on utilise juste la différence de xG
    xg_diff = df['xg_home'].values - df['xg_away'].values
    
    # 5. Optimisation
    result = optimize_coefficients(df, y, current_coeffs, method, verbose)
    
    if result['success']:
        # 6. Sauvegarde des nouveaux coefficients
        new_coeffs = result['coefficients']
        update_coefficients(new_coeffs)
        
        if verbose:
            print(f"\n✅ Nouveaux coefficients sauvegardés :")
            for k, v in new_coeffs.items():
                ancien = current_coeffs.get(k, 0)
                fleche = "↑" if v > ancien else "↓" if v < ancien else "="
                print(f"   {k}: {ancien:.4f} → {v:.4f} {fleche}")
        
        # 7. Évaluation rapide
        accuracy = evaluate_model(df, y, new_coeffs, verbose)
        result['accuracy'] = accuracy
        
        # 8. Calcul et stockage du facteur de confiance (inspiré par Benter)
        # Un Brier Score de 0.25 correspond au hasard. 0 est parfait.
        brier = accuracy.get('brier_score', 0.25)
        confidence = max(0.1, min(0.9, (0.25 - brier) / 0.25))
        set_config('model_confidence', str(round(confidence, 2)))
    
    if verbose:
        print("=" * 60 + "\n")
    
    return result


def optimize_coefficients(df, y, initial_coeffs, method='auto', verbose=True):
    """
    Optimise les coefficients par maximum de vraisemblance.
    
    Args:
        df: DataFrame avec les données
        y: cible (0/1)
        initial_coeffs: coefficients de départ
        method: 'full', 'simple', ou 'auto'
        verbose: affichage
    
    Returns:
        dict avec 'success', 'coefficients', 'log_likelihood', etc.
    """
    
    xg_diff = df['xg_home'].values - df['xg_away'].values
    
    # Déterminer la méthode
    nb_matchs = len(df)
    
    if method == 'auto':
        # Vérifier la diversité des stratégies dans l'historique
        strategy_counts = df.groupby(['strat_A', 'strat_B']).size()
        is_diverse = len(strategy_counts) >= 3 and strategy_counts.min() >= 2

        # Si moins de 20 matchs OU pas assez de diversité, on optimise seulement l'intercept.
        # Sinon, on optimise tous les coefficients.
        if nb_matchs < 20 or not is_diverse:
            method = 'simple'
            if verbose:
                reason = f"seulement {nb_matchs} matchs" if nb_matchs < 20 else "stratégies peu variées"
                print(f"\n📋 Méthode automatique → 'simple' ({reason})")
        else:
            method = 'full'
            if verbose:
                print(f"\n📋 Méthode automatique → 'full' ({nb_matchs} matchs)")
    
    if verbose:
        print(f"   Nombre de matchs utilisés : {nb_matchs}")
    
    if method == 'simple':
        return _optimize_intercept_only(xg_diff, y, initial_coeffs, verbose)
    else:
        return _optimize_all_coefficients(df, y, initial_coeffs, xg_diff, verbose)


def _optimize_intercept_only(xg_diff, y, initial_coeffs, verbose=True):
    """
    Optimise uniquement l'intercept du modèle.
    Les bonus tactiques restent inchangés.
    """
    
    if verbose:
        print("\n🔧 Optimisation de l'intercept uniquement...")
    
    # Paramètre initial
    initial_intercept = [initial_coeffs.get('intercept', 0.0)]
    
    # Optimisation
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                log_likelihood_simple,
                initial_intercept,
                args=(xg_diff, y.values),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000}
            )
        
        if result.success:
            new_coeffs = initial_coeffs.copy()
            new_coeffs['intercept'] = result.x[0]
            
            ll_initial = -log_likelihood_simple(initial_intercept, xg_diff, y.values)
            ll_final = -log_likelihood_simple(result.x, xg_diff, y.values)
            
            if verbose:
                print(f"   ✅ Optimisation réussie")
                print(f"   Log-vraisemblance : {ll_initial:.2f} → {ll_final:.2f}")
                print(f"   Intercept : {initial_intercept[0]:.4f} → {result.x[0]:.4f}")
            
            return {
                'success': True,
                'coefficients': new_coeffs,
                'log_likelihood_initial': ll_initial,
                'log_likelihood_final': ll_final,
                'iterations': result.nit,
                'method': 'simple'
            }
        else:
            if verbose:
                print(f"   ⚠️  Optimisation échouée : {result.message}")
            return {
                'success': False,
                'raison': result.message,
                'coefficients': initial_coeffs
            }
            
    except Exception as e:
        if verbose:
            print(f"   ❌ Erreur lors de l'optimisation : {e}")
        return {
            'success': False,
            'raison': str(e),
            'coefficients': initial_coeffs
        }


def _optimize_all_coefficients(df, y, initial_coeffs, xg_diff, verbose=True):
    """
    Optimise tous les coefficients (bonus tactiques + intercept).
    Nécessite suffisamment de données avec des stratégies variées.
    """
    
    if verbose:
        print("\n🔧 Optimisation de tous les coefficients (bonus tactiques + intercept)...")
    
    # Paramètres initiaux [bonus_ailes_pressing, bonus_ailes_bloc, 
    #                     bonus_axe_pressing, bonus_axe_bloc, intercept]
    initial_params = np.array([
        initial_coeffs.get('bonus_ailes_pressing', 0.20),
        initial_coeffs.get('bonus_ailes_bloc', 0.40),
        initial_coeffs.get('bonus_axe_pressing', 0.00),
        initial_coeffs.get('bonus_axe_bloc', 0.10),
        initial_coeffs.get('intercept', 0.00)
    ])
    
    # Bornes raisonnables pour les paramètres
    bounds = [
        (-0.5, 1.0),   # bonus_ailes_pressing
        (-0.5, 1.0),   # bonus_ailes_bloc
        (-0.5, 1.0),   # bonus_axe_pressing
        (-0.5, 1.0),   # bonus_axe_bloc
        (-1.0, 1.0)    # intercept
    ]
    
    def objective(params, lambda_reg=0.01):
        """Fonction objectif qui recalcule les features avec les nouveaux bonus."""
        coeffs = {
            'bonus_ailes_pressing': params[0],
            'bonus_ailes_bloc': params[1],
            'bonus_axe_pressing': params[2],
            'bonus_axe_bloc': params[3],
            'intercept': params[4]
        }
        
        # Recalculer les différences de xG ajustées avec les nouveaux bonus
        diff_adj = []
        for _, row in df.iterrows():
            bonus = calculer_bonus_tactique(
                row['strat_A'], row['strat_B'], coeffs
            )
            diff = row['xg_home'] - row['xg_away'] + bonus
            diff_adj.append(diff)
        
        diff_adj = np.array(diff_adj)
        
        # Log-vraisemblance
        z = diff_adj + params[4]
        probs = sigmoid(z)
        
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
        
        ll = y.values * np.log(probs) + (1 - y.values) * np.log(1 - probs)
        
        # Pénalité de régularisation L2 (sauf pour l'intercept)
        l2_penalty = lambda_reg * np.sum(params[:-1]**2)
        
        return -np.sum(ll) + l2_penalty
    
    # Optimisation
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Première tentative avec L-BFGS-B (avec bornes)
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-8}
            )
            
            # Si L-BFGS-B échoue, essayer Nelder-Mead
            if not result.success:
                if verbose:
                    print(f"   L-BFGS-B échoué ({result.message}), essai Nelder-Mead...")
                result = minimize(
                    objective,
                    initial_params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
                )
        
        if result.success or result.fun < objective(initial_params):
            new_coeffs = {
                'bonus_ailes_pressing': result.x[0],
                'bonus_ailes_bloc': result.x[1],
                'bonus_axe_pressing': result.x[2],
                'bonus_axe_bloc': result.x[3],
                'intercept': result.x[4]
            }
            
            ll_initial = -objective(initial_params)
            ll_final = -result.fun
            
            if verbose:
                print(f"   ✅ Optimisation réussie ({result.nit} itérations)")
                print(f"   Log-vraisemblance : {ll_initial:.2f} → {ll_final:.2f}")
                print(f"   Amélioration : {ll_final - ll_initial:.2f}")
            
            return {
                'success': True,
                'coefficients': new_coeffs,
                'log_likelihood_initial': ll_initial,
                'log_likelihood_final': ll_final,
                'iterations': result.nit,
                'method': 'full'
            }
        else:
            if verbose:
                print(f"   ⚠️  Optimisation échouée : {result.message}")
            return {
                'success': False,
                'raison': result.message,
                'coefficients': initial_coeffs
            }
            
    except Exception as e:
        if verbose:
            print(f"   ❌ Erreur lors de l'optimisation : {e}")
            traceback.print_exc()
        return {
            'success': False,
            'raison': str(e),
            'coefficients': initial_coeffs
        }


# ============================================================
# 4. ÉVALUATION DU MODÈLE
# ============================================================

def evaluate_model(df, y, coefficients, verbose=True, xg_diff_base=None):
    """
    Évalue la performance du modèle après recalibrage.
    
    Calcule :
    - Log-vraisemblance
    - Accuracy (seuil à 0.5)
    - Brier score (erreur quadratique moyenne)
    - Calibration (comparaison proba moyenne vs fréquence réelle)
    
    Args:
        df: DataFrame
        y: cible réelle
        coefficients: nouveaux coefficients
        xg_diff_base: Différence de xG pré-calculée (optimisation)
        verbose: affichage
    
    Returns:
        dict avec les métriques
    """
    
    # Prédictions
    if xg_diff_base is not None:
        # Mode simple, plus rapide
        z = xg_diff_base + coefficients.get('intercept', 0)
        predictions = sigmoid(z)
    else:
        # Mode complet, nécessite de recalculer les bonus
        predictions = []
        for _, row in df.iterrows():
            bonus = calculer_bonus_tactique(
                row.get('strat_A', 'ailes'), row.get('strat_B', 'pressing_haut'), coefficients
            )
            diff_adj = row['xg_home'] - row['xg_away'] + bonus
            z = diff_adj + coefficients.get('intercept', 0)
            prob = sigmoid(z)
            predictions.append(prob)
    
    predictions = np.array(predictions)
    y_true = y.values
    
    # Métriques
    # Accuracy (seuil 0.5)
    y_pred_binary = (predictions >= 0.5).astype(int)
    accuracy = np.mean(y_pred_binary == y_true)
    
    # Brier score (erreur quadratique moyenne)
    brier_score = np.mean((predictions - y_true) ** 2)
    
    # Log-vraisemblance moyenne
    epsilon = 1e-15
    probs_clipped = np.clip(predictions, epsilon, 1 - epsilon)
    log_loss = -np.mean(y_true * np.log(probs_clipped) + 
                         (1 - y_true) * np.log(1 - probs_clipped))
    
    # Calibration : comparer proba moyenne et fréquence réelle
    prob_moyenne = np.mean(predictions)
    freq_reelle = np.mean(y_true)
    
    if verbose:
        print(f"\n📊 ÉVALUATION DU MODÈLE :")
        print(f"   Accuracy (seuil 0.5) : {accuracy:.1%}")
        print(f"   Brier score : {brier_score:.4f} (0 = parfait, 0.25 = aléatoire)")
        print(f"   Log-loss : {log_loss:.4f}")
        print(f"   Probabilité moyenne prédite : {prob_moyenne:.1%}")
        print(f"   Fréquence réelle de victoire domicile : {freq_reelle:.1%}")
        print(f"   Écart de calibration : {prob_moyenne - freq_reelle:+.1%}")
        
        # Interprétation
        if abs(prob_moyenne - freq_reelle) < 0.05:
            print(f"   ✅ Bonne calibration")
        elif prob_moyenne > freq_reelle:
            print(f"   ⚠️  Le modèle surestime les victoires à domicile")
        else:
            print(f"   ⚠️  Le modèle sous-estime les victoires à domicile")
    
    return {
        'accuracy': accuracy,
        'brier_score': brier_score,
        'log_loss': log_loss,
        'prob_moyenne': prob_moyenne,
        'freq_reelle': freq_reelle,
        'nb_matchs': len(df)
    }


# ============================================================
# 5. VALIDATION CROISÉE
# ============================================================

def cross_validate(n_folds=5, method='auto', verbose=True):
    """
    Validation croisée pour évaluer la robustesse du modèle.
    
    Args:
        n_folds: nombre de plis
        verbose: affichage
    
    Returns:
        dict avec les métriques moyennes
    """
    
    df, y = load_training_data()
    
    if df is None or len(df) < n_folds * 2:
        print("⚠️  Pas assez de données pour la validation croisée.")
        return None
    
    df = extract_strategies(df)
    
    # Si la méthode est 'auto', on la détermine une seule fois sur l'ensemble des données
    if method == 'auto':
        method = 'full' if len(df) >= 20 and df.groupby(['strat_A', 'strat_B']).size().min() >= 2 else 'simple'
    
    # Mélanger les données
    indices = np.random.permutation(len(df))
    fold_size = len(df) // n_folds
    
    metrics = {
        'accuracy': [],
        'brier_score': [],
        'log_loss': []
    }
    
    if verbose:
        print(f"\n🔬 VALIDATION CROISÉE ({n_folds} plis)")
        print("-" * 40)
    
    for fold in range(n_folds):
        # Séparation train/test
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(df)
        
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        
        df_train, y_train = df.iloc[train_idx], y.iloc[train_idx]
        df_test, y_test = df.iloc[test_idx], y.iloc[test_idx]
        
        # Entraînement sur le pli
        initial_coeffs = get_coefficients()
        result = optimize_coefficients(df_train, y_train, initial_coeffs, method=method, verbose=False)
        
        if result['success']:
            # Évaluation sur le test
            eval_result = evaluate_model(df_test, y_test, result['coefficients'], verbose=False)
            
            metrics['accuracy'].append(eval_result['accuracy'])
            metrics['brier_score'].append(eval_result['brier_score'])
            metrics['log_loss'].append(eval_result['log_loss'])
            
            if verbose:
                print(f"   Pli {fold+1}/{n_folds} : accuracy={eval_result['accuracy']:.1%}, "
                      f"brier={eval_result['brier_score']:.4f}")
    
    if verbose and metrics['accuracy']:
        print("-" * 60)
        print(f"   📊 RÉSULTATS MOYENS (méthode '{method}')")
        print(f"   Accuracy : {np.mean(metrics['accuracy']):.1%} (±{np.std(metrics['accuracy']):.1%})")
        print(f"   Brier    : {np.mean(metrics['brier_score']):.4f} (±{np.std(metrics['brier_score']):.4f})")
        print(f"   Log-loss : {np.mean(metrics['log_loss']):.4f} (±{np.std(metrics['log_loss']):.4f})")
        print("-" * 60)
    
    return {
        'accuracy_mean': np.mean(metrics['accuracy']) if metrics['accuracy'] else 0,
        'accuracy_std': np.std(metrics['accuracy']) if metrics['accuracy'] else 0,
        'brier_mean': np.mean(metrics['brier_score']) if metrics['brier_score'] else 0,
        'log_loss_mean': np.mean(metrics['log_loss']) if metrics['log_loss'] else 0,
        'nb_folds': n_folds
    }


# ============================================================
# 6. RAPPORT DE RECALIBRAGE
# ============================================================

def get_recalibration_report():
    """
    Génère un rapport complet sur l'état du recalibrage.
    
    Returns:
        dict avec toutes les informations
    """
    report = {
        'date': datetime.now().isoformat(),
        'coefficients_actuels': get_coefficients(),
        'nb_matchs_historique': get_nb_matchs_historique(),
        'dernier_recalibrage': None
    }
    
    # Récupérer la date du dernier recalibrage
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT derniere_maj FROM coefficients WHERE id=1")
        row = cursor.fetchone()
        if row and row[0]:
            report['dernier_recalibrage'] = row[0]
        conn.close()
    except:
        pass
    
    # Évaluation sur les données actuelles
    df, y = load_training_data()
    if df is not None and len(df) > 0:
        df = extract_strategies(df)
        eval_result = evaluate_model(df, y, report['coefficients_actuels'], verbose=False)
        report['evaluation'] = eval_result
    
    return report


# ============================================================
# 7. TEST
# ============================================================

if __name__ == "__main__":
    print("🧪 Test du module recalibrator\n")
    
    # Initialisation base de données
    from db import init_db
    init_db()
    
    # Ajouter des données de test si la base est vide
    nb_matchs = get_nb_matchs_historique()
    
    if nb_matchs < 20:
        print("📝 Ajout de données de test pour la démonstration...")
        from db import ajouter_match_historique, mettre_a_jour_resultat
        import random
        
        random.seed(42)
        
        equipes = [
            ("Arsenal", "Chelsea"), ("Liverpool", "Man United"),
            ("Man City", "Tottenham"), ("Newcastle", "Aston Villa"),
            ("Brighton", "West Ham"), ("PSG", "Marseille"),
            ("Lyon", "Monaco"), ("Lille", "Rennes")
        ]
        
        strategies_possibles = [
            ("ailes", "pressing_haut"),
            ("ailes", "bloc_bas"),
            ("axe", "pressing_haut"),
            ("axe", "bloc_bas")
        ]
        
        for i in range(30):
            home, away = random.choice(equipes)
            strat_A, strat_B = random.choice(strategies_possibles)
            
            xg_home = 1.5 + random.uniform(-0.5, 1.0)
            xg_away = 1.3 + random.uniform(-0.3, 0.8)
            
            # Déterminer le résultat en fonction des xG
            diff = xg_home - xg_away
            if diff > 0.5:
                resultat = 'home'
            elif diff < -0.3:
                resultat = 'away'
            else:
                resultat = random.choice(['home', 'away', 'draw'])
            
            match_id = ajouter_match_historique(
                home_team=home,
                away_team=away,
                date_match=datetime.now().isoformat(),
                xg_home=round(xg_home, 2),
                xg_away=round(xg_away, 2),
                possession_home=random.uniform(40, 65),
                possession_away=random.uniform(35, 60),
                strategies={"A": strat_A, "B": strat_B},
                prob_estimee_home=0.5,
                cote_home=2.0,
                cote_draw=3.5,
                cote_away=3.5
            )
            
            mettre_a_jour_resultat(match_id, resultat)
        
        print(f"   {30} matchs de test ajoutés.\n")
    
    # Test du recalibrage
    print("1. Test recalibrage automatique :")
    resultat = recalibrate(method='auto', verbose=True)
    
    print("\n2. Rapport de recalibrage :")
    rapport = get_recalibration_report()
    for k, v in rapport.items():
        if k != 'evaluation' and k != 'coefficients_actuels':
            print(f"   {k}: {v}")
    
    print("\n3. Test validation croisée :")
    cv_result = cross_validate(n_folds=5, verbose=True)
    
    print("\n✅ Tests terminés !")