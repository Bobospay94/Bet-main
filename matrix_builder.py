"""
Matrix Builder - Construction de la matrice de gains
=====================================================

Ce module transforme les cotes et les choix tactiques en une matrice 
de probabilités de victoire pour le jeu à somme nulle.

Compatible avec :
- The Odds API seule (estimation des xG depuis les cotes)
- API-Football (utilise les vrais xG si disponibles)
- Mode simulation (données de démonstration)

La matrice est au format :
    Équipe B →
Équipe A    stratégie_B1  stratégie_B2
↓
stratégie_A1   p(win|A1,B1)   p(win|A1,B2)
stratégie_A2   p(win|A2,B1)   p(win|A2,B2)
"""

import numpy as np
import math
from datetime import datetime


# ============================================================
# 1. ESTIMATION DES XG DEPUIS LES COTES (NOUVEAU)
# ============================================================

def estimer_xg_depuis_cotes(cote_home, cote_draw, cote_away):
    """
    Estime les expected goals (xG) à partir des cotes 1X2.
    
    Principe : Les cotes des bookmakers sont le meilleur indicateur 
    de la force réelle des équipes. On peut en déduire des xG cohérents.
    
    Méthode :
    1. On extrait les probabilités implicites des cotes
    2. On corrige la marge du bookmaker (overround)
    3. On convertit les probabilités en xG via une formule empirique
    
    Args:
        cote_home: Cote victoire domicile (ex: 2.10)
        cote_draw: Cote match nul (ex: 3.50)
        cote_away: Cote victoire extérieur (ex: 3.40)
    
    Returns:
        dict avec xG_for, xG_against, possession, forme
    """
    
    if not cote_home or not cote_draw or not cote_away:
        # Valeurs par défaut si cotes manquantes
        return {
            'xG_for': 1.5,
            'xG_against': 1.3,
            'possession': 50.0,
            'forme': ''
        }
    
    # Étape 1 : Probabilités implicites brutes
    p_home_brute = 1.0 / cote_home
    p_draw_brute = 1.0 / cote_draw
    p_away_brute = 1.0 / cote_away
    
    # Étape 2 : Calcul de la marge du bookmaker (overround)
    marge = p_home_brute + p_draw_brute + p_away_brute - 1.0
    
    # Étape 3 : Probabilités corrigées (sans la marge)
    # On répartit la marge proportionnellement
    p_home = p_home_brute / (1.0 + marge)
    p_draw = p_draw_brute / (1.0 + marge)
    p_away = p_away_brute / (1.0 + marge)
    
    # Étape 4 : Conversion probabilités → xG
    # Formule empirique calibrée sur les grands championnats
    # Une équipe avec 70% de chances de gagner → ~2.2 xG
    # Une équipe avec 50% de chances de gagner → ~1.5 xG
    # Une équipe avec 30% de chances de gagner → ~1.0 xG
    # Une équipe avec 15% de chances de gagner → ~0.7 xG
    
    xg_home = 0.4 + p_home * 2.6 - p_draw * 0.3
    xg_away = 0.4 + p_away * 2.6 - p_draw * 0.3
    
    # Ajustement pour que le xG total soit cohérent
    # Le total moyen de xG par match est d'environ 2.8
    total_xg = xg_home + xg_away
    if total_xg > 0:
        facteur_normalisation = 2.8 / total_xg
        xg_home *= facteur_normalisation
        xg_away *= facteur_normalisation
    
    # Limiter les valeurs aberrantes
    xg_home = max(0.3, min(3.5, xg_home))
    xg_away = max(0.3, min(3.5, xg_away))
    
    # Étape 5 : Estimation de la possession
    # L'équipe favorite a généralement plus de possession
    possession = 50.0 + (p_home - p_away) * 25.0
    possession = max(30.0, min(70.0, possession))
    
    return {
        'xG_for': round(xg_home, 2),
        'xG_against': round(xg_away, 2),  # xG contre = xG de l'adversaire
        'possession': round(possession, 1),
        'forme': ''  # La forme sera calculée séparément si disponible
    }


def estimer_stats_depuis_cotes(cote_home, cote_draw, cote_away):
    """
    Version complète qui retourne les stats pour les deux équipes.
    
    Returns:
        tuple: (stats_home, stats_away)
    """
    stats_home = estimer_xg_depuis_cotes(cote_home, cote_draw, cote_away)
    
    # Pour l'équipe extérieur, on inverse
    stats_away = estimer_xg_depuis_cotes(cote_away, cote_draw, cote_home)
    
    return stats_home, stats_away


# ============================================================
# 2. CONVERSION xG → PROBABILITÉ DE VICTOIRE
# ============================================================

def xg_to_win_probability(xg_home, xg_away, method="logistic"):
    """
    Convertit des expected goals (xG) en probabilité de victoire à domicile.
    
    Méthodes disponibles :
    - "logistic" : fonction logistique basée sur la différence de xG (recommandé)
    - "poisson"  : simulation basée sur la loi de Poisson (plus réaliste mais lent)
    - "empirique" : basée sur une formule calibrée sur données réelles
    
    Args:
        xg_home: Expected goals de l'équipe à domicile
        xg_away: Expected goals de l'équipe à l'extérieur
        method: Méthode de conversion
    
    Returns:
        float: Probabilité de victoire à domicile (entre 0 et 1)
    """
    
    if method == "logistic":
        # Fonction logistique simple
        # Plus la différence de xG est grande, plus la probabilité est élevée
        # Un écart de 0.5 xG donne environ 62% de chances
        diff = xg_home - xg_away
        k = 2.5  # Paramètre de pente (plus k est grand, plus c'est sensible)
        return 1.0 / (1.0 + math.exp(-k * diff))
    
    elif method == "poisson":
        # Simulation Poisson : plus précise mais plus lourde
        # On simule 5000 matchs et on compte les victoires
        np.random.seed(42)
        n_simulations = 5000
        
        goals_home = np.random.poisson(lam=xg_home, size=n_simulations)
        goals_away = np.random.poisson(lam=xg_away, size=n_simulations)
        
        home_wins = np.sum(goals_home > goals_away)
        prob_home_win = home_wins / n_simulations
        
        return prob_home_win
    
    elif method == "empirique":
        # Formule calibrée sur les données réelles
        # Basée sur la relation observée entre xG et résultats dans le football
        diff = xg_home - xg_away
        
        if diff > 2.0:
            return 0.88
        elif diff > 1.5:
            return 0.78
        elif diff > 1.0:
            return 0.68
        elif diff > 0.5:
            return 0.58
        elif diff > 0.0:
            return 0.48
        elif diff > -0.5:
            return 0.38
        elif diff > -1.0:
            return 0.28
        elif diff > -1.5:
            return 0.18
        elif diff > -2.0:
            return 0.10
        else:
            return 0.06
    
    else:
        raise ValueError(f"Méthode inconnue : {method}")


def xg_to_draw_probability(xg_home, xg_away):
    """
    Estime la probabilité de match nul à partir des xG.
    Utilise la simulation Poisson pour plus de précision.
    """
    n_simulations = 5000
    np.random.seed(42)
    
    goals_home = np.random.poisson(lam=xg_home, size=n_simulations)
    goals_away = np.random.poisson(lam=xg_away, size=n_simulations)
    
    draws = np.sum(goals_home == goals_away)
    return draws / n_simulations


# ============================================================
# 3. AJUSTEMENTS TACTIQUES
# ============================================================

def calculer_bonus_tactique(strategie_A, strategie_B, coefficients):
    """
    Calcule le bonus/malus de xG selon l'interaction tactique.
    
    Stratégies disponibles :
    - Équipe A : "ailes" (jeu large, centres) ou "axe" (jeu dans l'axe, combinaisons)
    - Équipe B : "pressing_haut" (défense haute, pressing) ou "bloc_bas" (défense basse, contre)
    
    Interactions typiques :
    - Ailes vs Pressing haut : bonus modéré (espaces derrière le pressing)
    - Ailes vs Bloc bas : bonus fort (centres dans une défense regroupée)  
    - Axe vs Pressing haut : bonus faible (milieu verrouillé)
    - Axe vs Bloc bas : bonus modéré (combinaisons dans la surface)
    
    Args:
        strategie_A: "ailes" ou "axe"
        strategie_B: "pressing_haut" ou "bloc_bas"
        coefficients: dict avec les bonus tactiques
    
    Returns:
        float: bonus de xG pour l'équipe A
    """
    
    # Valeurs par défaut si coefficients incomplets
    defaults = {
        "bonus_ailes_pressing": 0.20,
        "bonus_ailes_bloc": 0.40,
        "bonus_axe_pressing": 0.00,
        "bonus_axe_bloc": 0.10
    }
    
    mapping = {
        ("ailes", "pressing_haut"): coefficients.get("bonus_ailes_pressing", defaults["bonus_ailes_pressing"]),
        ("ailes", "bloc_bas"):      coefficients.get("bonus_ailes_bloc", defaults["bonus_ailes_bloc"]),
        ("axe",  "pressing_haut"):  coefficients.get("bonus_axe_pressing", defaults["bonus_axe_pressing"]),
        ("axe",  "bloc_bas"):       coefficients.get("bonus_axe_bloc", defaults["bonus_axe_bloc"]),
    }
    
    return mapping.get((strategie_A, strategie_B), 0.0)


def ajuster_xg_selon_forme(xg_base, forme):
    """
    Ajuste les xG selon la forme récente de l'équipe.
    
    Args:
        xg_base: xG de base
        forme: chaîne comme "WWDLW" (W=win, D=draw, L=loss)
    
    Returns:
        float: xG ajusté
    """
    if not forme or len(forme) == 0:
        return xg_base
    
    # Score de forme : W=+0.15, D=0, L=-0.15
    score = 0
    for i, char in enumerate(forme.upper()):
        poids = 1.0 - (i * 0.1)  # Les matchs récents comptent plus
        if char == 'W':
            score += 0.15 * poids
        elif char == 'L':
            score -= 0.15 * poids
    
    # Normaliser
    score /= len(forme)
    
    return max(0.3, xg_base + score)


# ============================================================
# 4. CONSTRUCTION DE LA MATRICE DE GAINS (VERSION PRINCIPALE)
# ============================================================

def build_payoff_matrix(team_A_stats, team_B_stats, strategies_A, strategies_B, 
                         coefficients=None, method="logistic"):
    """
    Construit la matrice de gains 2×2 pour le jeu à somme nulle.
    
    COMPATIBLE AVEC :
    - Stats réelles (API-Football)
    - Stats estimées depuis les cotes (The Odds API seule)
    - Stats de démonstration
    
    Args:
        team_A_stats: dict avec 'xG_for', 'xG_against', 'possession', 'forme'
        team_B_stats: dict avec 'xG_for', 'xG_against', 'possession', 'forme'
        strategies_A: liste de 2 stratégies, ex: ["ailes", "axe"]
        strategies_B: liste de 2 stratégies, ex: ["pressing_haut", "bloc_bas"]
        coefficients: dict des coefficients tactiques (None = valeurs par défaut)
        method: "logistic", "poisson", ou "empirique"
    
    Returns:
        numpy.array 2×2 : probabilité que A gagne pour chaque couple
    """
    
    # Coefficients par défaut
    if coefficients is None:
        coefficients = {
            "bonus_ailes_pressing": 0.20,
            "bonus_ailes_bloc": 0.40,
            "bonus_axe_pressing": 0.00,
            "bonus_axe_bloc": 0.10,
            "intercept": 0.00
        }
    
    # Récupération des statistiques avec valeurs par défaut
    xg_for_A = team_A_stats.get('xG_for', 1.5)
    xg_against_A = team_A_stats.get('xG_against', 1.3)
    possession_A = team_A_stats.get('possession', 50)
    forme_A = team_A_stats.get('forme', '')
    
    xg_for_B = team_B_stats.get('xG_for', 1.5)
    xg_against_B = team_B_stats.get('xG_against', 1.3)
    possession_B = team_B_stats.get('possession', 50)
    forme_B = team_B_stats.get('forme', '')
    
    # Ajustement selon la forme
    xg_for_A = ajuster_xg_selon_forme(xg_for_A, forme_A)
    xg_for_B = ajuster_xg_selon_forme(xg_for_B, forme_B)
    
    # Construction de la matrice 2×2
    matrice = np.zeros((2, 2))
    
    for i, strat_A in enumerate(strategies_A):
        for j, strat_B in enumerate(strategies_B):
            
            # Bonus tactique pour cette interaction
            bonus = calculer_bonus_tactique(strat_A, strat_B, coefficients)
            
            # xG effectif de l'équipe A
            # = son xG offensif + bonus tactique - qualité défensive de B
            xg_A_effectif = xg_for_A + bonus - (xg_against_B - 1.0) * 0.4
            
            # xG effectif de l'équipe B
            # = son xG offensif - qualité défensive de A
            xg_B_effectif = xg_for_B - (xg_against_A - 1.0) * 0.4
            
            # Application de l'intercept (biais global du modèle)
            xg_A_effectif += coefficients.get("intercept", 0.0)
            
            # Ajustement possession (impact léger)
            ecart_possession = (possession_A - possession_B) / 100.0
            xg_A_effectif *= (1.0 + ecart_possession * 0.3)
            xg_B_effectif *= (1.0 - ecart_possession * 0.3)
            
            # Limiter à des valeurs raisonnables
            xg_A_effectif = max(0.2, min(4.0, xg_A_effectif))
            xg_B_effectif = max(0.2, min(4.0, xg_B_effectif))
            
            # Conversion en probabilité
            prob_victoire_A = xg_to_win_probability(xg_A_effectif, xg_B_effectif, method)
            
            matrice[i, j] = prob_victoire_A
    
    return matrice


# ============================================================
# 5. VERSION ÉTENDUE AVEC DÉTAILS
# ============================================================

def build_extended_payoff_matrix(team_A_stats, team_B_stats, 
                                  strategies_A, strategies_B, coefficients=None):
    """
    Version étendue qui retourne plus d'informations sur chaque case.
    
    Returns:
        dict avec :
        - 'matrice' : numpy.array 2×2
        - 'details' : DataFrame avec xG effectifs et probabilités détaillées
        - 'stats_ajustees' : statistiques après ajustements
    """
    import pandas as pd
    
    if coefficients is None:
        coefficients = {
            "bonus_ailes_pressing": 0.20,
            "bonus_ailes_bloc": 0.40,
            "bonus_axe_pressing": 0.00,
            "bonus_axe_bloc": 0.10,
            "intercept": 0.00
        }
    
    xg_for_A = team_A_stats.get('xG_for', 1.5)
    xg_against_A = team_A_stats.get('xG_against', 1.3)
    possession_A = team_A_stats.get('possession', 50)
    forme_A = team_A_stats.get('forme', '')
    
    xg_for_B = team_B_stats.get('xG_for', 1.5)
    xg_against_B = team_B_stats.get('xG_against', 1.3)
    possession_B = team_B_stats.get('possession', 50)
    forme_B = team_B_stats.get('forme', '')
    
    xg_for_A = ajuster_xg_selon_forme(xg_for_A, forme_A)
    xg_for_B = ajuster_xg_selon_forme(xg_for_B, forme_B)
    
    matrice = np.zeros((2, 2))
    details = []
    
    for i, strat_A in enumerate(strategies_A):
        for j, strat_B in enumerate(strategies_B):
            bonus = calculer_bonus_tactique(strat_A, strat_B, coefficients)
            
            xg_A_eff = xg_for_A + bonus - (xg_against_B - 1.0) * 0.4
            xg_B_eff = xg_for_B - (xg_against_A - 1.0) * 0.4
            xg_A_eff += coefficients.get("intercept", 0.0)
            
            ecart_possession = (possession_A - possession_B) / 100.0
            xg_A_eff *= (1.0 + ecart_possession * 0.3)
            xg_B_eff *= (1.0 - ecart_possession * 0.3)
            
            xg_A_eff = max(0.2, min(4.0, xg_A_eff))
            xg_B_eff = max(0.2, min(4.0, xg_B_eff))
            
            prob_A = xg_to_win_probability(xg_A_eff, xg_B_eff, "logistic")
            prob_draw = xg_to_draw_probability(xg_A_eff, xg_B_eff)
            prob_B = 1.0 - prob_A - prob_draw
            
            matrice[i, j] = prob_A
            
            details.append({
                'strat_A': strat_A,
                'strat_B': strat_B,
                'bonus_tactique': round(bonus, 3),
                'xg_A_effectif': round(xg_A_eff, 2),
                'xg_B_effectif': round(xg_B_eff, 2),
                'prob_victoire_A': round(prob_A, 3),
                'prob_nul': round(prob_draw, 3),
                'prob_victoire_B': round(prob_B, 3)
            })
    
    return {
        'matrice': matrice,
        'details': pd.DataFrame(details),
        'stats_ajustees': {
            'A': {
                'xg_for': round(xg_for_A, 2),
                'xg_against': round(xg_against_A, 2),
                'possession': round(possession_A, 1)
            },
            'B': {
                'xg_for': round(xg_for_B, 2),
                'xg_against': round(xg_against_B, 2),
                'possession': round(possession_B, 1)
            }
        }
    }


# ============================================================
# 6. VERSION SIMPLIFIÉE (sans tactiques)
# ============================================================

def build_payoff_matrix_simple(xg_home, xg_away, home_advantage=0.2):
    """
    Version simplifiée sans stratégies tactiques.
    Utile pour les tests rapides ou les cas simples.
    
    Args:
        xg_home: xG moyen de l'équipe à domicile
        xg_away: xG moyen de l'équipe à l'extérieur
        home_advantage: bonus de xG pour l'équipe à domicile
    
    Returns:
        numpy.array 2×2 (identique sur chaque case)
    """
    xg_home_adj = xg_home + home_advantage
    prob = xg_to_win_probability(xg_home_adj, xg_away)
    
    return np.array([[prob, prob],
                     [prob, prob]])


# ============================================================
# 7. AFFICHAGE
# ============================================================

def afficher_matrice(matrice, strategies_A, strategies_B, 
                     equipe_A="Équipe A", equipe_B="Équipe B"):
    """
    Affiche la matrice de gains de manière lisible dans la console.
    """
    lignes = []
    lignes.append(f"\n{'='*60}")
    lignes.append(f"  MATRICE DE GAINS : Probabilité que {equipe_A} gagne")
    lignes.append(f"{'='*60}")
    lignes.append(f"\n  {equipe_B} →")
    lignes.append(f"  {equipe_A} ↓")
    lignes.append(f"  {'':20} {strategies_B[0]:20} {strategies_B[1]:20}")
    lignes.append(f"  {'-'*60}")
    
    for i, strat_A in enumerate(strategies_A):
        valeurs = f"  {strat_A:20} "
        for j in range(2):
            valeurs += f"{matrice[i, j]:.1%}                "
        lignes.append(valeurs)
    
    lignes.append(f"{'='*60}\n")
    return "\n".join(lignes)


def matrice_to_dataframe(matrice, strategies_A, strategies_B):
    """
    Convertit la matrice en DataFrame pandas pour affichage Streamlit.
    """
    import pandas as pd
    
    df = pd.DataFrame(
        matrice,
        index=[f"A: {s}" for s in strategies_A],
        columns=[f"B: {s}" for s in strategies_B]
    )
    # Formater en pourcentages
    df = df.applymap(lambda x: f"{x:.1%}")
    return df


# ============================================================
# 8. UTILITAIRE POUR LE RECALIBRAGE
# ============================================================

def calculer_xg_effectif(stats_home, stats_away, strat_A, strat_B, coefficients):
    """
    Calcule les xG effectifs pour un couple de stratégies.
    Utilisé par le recalibrator pour la fonction de vraisemblance.
    
    Returns:
        tuple: (xg_A_effectif, xg_B_effectif)
    """
    xg_for_A = stats_home.get('xG_for', 1.5)
    xg_against_A = stats_home.get('xG_against', 1.3)
    xg_for_B = stats_away.get('xG_for', 1.5)
    xg_against_B = stats_away.get('xG_against', 1.3)
    
    bonus = calculer_bonus_tactique(strat_A, strat_B, coefficients)
    
    xg_A_eff = xg_for_A + bonus - (xg_against_B - 1.0) * 0.4
    xg_B_eff = xg_for_B - (xg_against_A - 1.0) * 0.4
    xg_A_eff += coefficients.get("intercept", 0.0)
    
    xg_A_eff = max(0.2, min(4.0, xg_A_eff))
    xg_B_eff = max(0.2, min(4.0, xg_B_eff))
    
    return xg_A_eff, xg_B_eff


# ============================================================
# 9. TEST
# ============================================================

if __name__ == "__main__":
    print("🧪 Test du module matrix_builder (version The Odds API)\n")
    
    # Test 1 : Estimation xG depuis cotes
    print("=" * 60)
    print("TEST 1 : Estimation des xG depuis les cotes")
    print("=" * 60)
    
    test_cotes = [
        (2.10, 3.50, 3.40),   # Match équilibré avec léger favori
        (1.50, 4.00, 7.00),   # Favori net
        (3.00, 3.20, 2.40),   # Léger avantage extérieur
        (1.20, 6.00, 12.00),  # Très gros favori
    ]
    
    for cote_h, cote_d, cote_a in test_cotes:
        stats_h, stats_a = estimer_stats_depuis_cotes(cote_h, cote_d, cote_a)
        print(f"\n   Cotes : {cote_h:.2f} / {cote_d:.2f} / {cote_a:.2f}")
        print(f"   Domicile : xG={stats_h['xG_for']}, xGA={stats_h['xG_against']}, "
              f"Poss={stats_h['possession']}%")
        print(f"   Extérieur : xG={stats_a['xG_for']}, xGA={stats_a['xG_against']}, "
              f"Poss={stats_a['possession']}%")
    
    # Test 2 : Construction de la matrice
    print("\n" + "=" * 60)
    print("TEST 2 : Matrice de gains complète")
    print("=" * 60)
    
    # Simuler des stats (comme si venant des cotes)
    team_A = {'xG_for': 1.9, 'xG_against': 1.0, 'possession': 55, 'forme': 'WWDLW'}
    team_B = {'xG_for': 1.3, 'xG_against': 1.4, 'possession': 45, 'forme': 'LWDLL'}
    
    strategies_A = ["ailes", "axe"]
    strategies_B = ["pressing_haut", "bloc_bas"]
    
    coefficients = {
        "bonus_ailes_pressing": 0.20,
        "bonus_ailes_bloc": 0.40,
        "bonus_axe_pressing": 0.00,
        "bonus_axe_bloc": 0.10,
        "intercept": 0.00
    }
    
    mat = build_payoff_matrix(team_A, team_B, strategies_A, strategies_B, coefficients)
    print(afficher_matrice(mat, strategies_A, strategies_B, "Arsenal", "Chelsea"))
    
    # Test 3 : Version étendue
    print("=" * 60)
    print("TEST 3 : Version étendue avec détails")
    print("=" * 60)
    
    result = build_extended_payoff_matrix(team_A, team_B, strategies_A, strategies_B, coefficients)
    print(result['details'].to_string())
    
    # Test 4 : Différentes méthodes de conversion
    print("\n" + "=" * 60)
    print("TEST 4 : Comparaison des méthodes de conversion xG")
    print("=" * 60)
    
    print(f"\n   {'xG Home':>8} {'xG Away':>8} {'Logistic':>10} {'Poisson':>10} {'Empirique':>10}")
    print("   " + "-" * 50)
    
    for diff in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
        xg_h = 1.5 + diff/2
        xg_a = 1.5 - diff/2
        prob_log = xg_to_win_probability(xg_h, xg_a, "logistic")
        prob_poi = xg_to_win_probability(xg_h, xg_a, "poisson")
        prob_emp = xg_to_win_probability(xg_h, xg_a, "empirique")
        print(f"   {xg_h:>8.2f} {xg_a:>8.2f} {prob_log:>10.1%} {prob_poi:>10.1%} {prob_emp:>10.1%}")
    
    # Test 5 : Ajustement selon forme
    print("\n" + "=" * 60)
    print("TEST 5 : Ajustement selon forme récente")
    print("=" * 60)
    
    for forme in ['WWWWW', 'WWDLW', 'WDLWL', 'LWDLL', 'LLLLL']:
        xg_adj = ajuster_xg_selon_forme(1.5, forme)
        print(f"   Forme '{forme}' → xG ajusté: {xg_adj:.2f} (base: 1.50)")
    
    # Test 6 : Bonus tactiques
    print("\n" + "=" * 60)
    print("TEST 6 : Bonus tactiques par interaction")
    print("=" * 60)
    
    for strat_A in ["ailes", "axe"]:
        for strat_B in ["pressing_haut", "bloc_bas"]:
            bonus = calculer_bonus_tactique(strat_A, strat_B, coefficients)
            print(f"   {strat_A:15} vs {strat_B:15} → bonus xG: {bonus:+.2f}")
    
    print("\n✅ Tests terminés !")