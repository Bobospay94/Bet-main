"""
Value Detector - Détection de Value Bets et Gestion des Mises
===============================================================

Ce module est le cœur décisionnel du système. Il :
1. Compare les probabilités estimées aux cotes des bookmakers
2. Détecte les situations de value bet (espérance positive)
3. Calcule la mise optimale selon le critère de Kelly
4. Gère le bankroll et applique des règles de prudence
5. Filtre les opportunités selon des critères de qualité

Le critère de Kelly :
    f* = (p * b - q) / b
    où f* = fraction du capital à miser
         p  = probabilité estimée de gagner
         b  = cote décimale - 1 (gain net par unité misée)
         q  = 1 - p (probabilité de perdre)

Le Kelly fractionnel (plus prudent) :
    f = f* * fraction
    où fraction est typiquement 0.25 (quart de Kelly)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import sqlite3
import math
from itertools import combinations

# Imports locaux
from db import get_config, set_config, get_statistiques_paris, get_paris_termines


# ============================================================
# 1. CALCUL DE LA VALUE ET DU CRITÈRE DE KELLY
# ============================================================

def calculate_expected_value(prob_est, cote, confidence_factor=None):
    """
    Calcule l'espérance de gain (Expected Value) d'un pari.
    
    Inspiré par Bill Benter : on mélange l'estimation du modèle avec 
    la probabilité du marché pour réduire la variance.
    
    Args:
        prob_est: Notre probabilité estimée (entre 0 et 1)
        cote: Cote décimale du bookmaker
        confidence_factor: Confiance en notre modèle (0.0 à 1.0)
                           0.7 signifie 70% modèle, 30% marché.
    
    Returns:
        float: Espérance de gain (ex: 0.05 = 5% d'avantage)
    """
    if confidence_factor is None:
        try:
            val = get_config('model_confidence')
            confidence_factor = float(val) if val else 0.7
        except:
            confidence_factor = 0.7

    if cote <= 1.0 or prob_est <= 0 or prob_est >= 1:
        return 0.0
    
    # Probabilité implicite du marché (bookmaker)
    prob_market = 1.0 / cote
    
    # Probabilité combinée (Weighted Average)
    p_adj = (prob_est * confidence_factor) + (prob_market * (1.0 - confidence_factor))
    
    return p_adj * cote - 1


def calculate_kelly_fraction(prob_est, cote, fraction=0.25, bankroll=None):
    """
    Calcule la fraction optimale du capital à miser selon Kelly.
    
    Formule de Kelly complète :
        f* = (p * b - (1-p)) / b
        où b = cote - 1
    
    Args:
        prob_est: Probabilité estimée de gagner
        cote: Cote décimale
        fraction: Fraction de Kelly à appliquer (0.25 = quart de Kelly)
        bankroll: Capital total (optionnel, pour calculer la mise en euros)
    
    Returns:
        float: Fraction du capital à miser (entre 0 et 1)
               ou montant en euros si bankroll est fourni
    """
    
    # Validation des entrées
    if cote <= 1.0:
        return 0.0
    if prob_est <= 0 or prob_est >= 1:
        return 0.0
    
    # Calcul du critère de Kelly
    b = cote - 1  # Gain net
    p = prob_est
    q = 1 - p
    
    # Kelly complet
    kelly_full = (p * b - q) / b
    
    # Si Kelly négatif, ne pas parier
    if kelly_full <= 0:
        return 0.0
    
    # Appliquer la fraction de prudence
    kelly_fractional = kelly_full * fraction
    
    # Limiter la mise maximale (sécurité)
    kelly_fractional = min(kelly_fractional, 0.10)  # Max 10% du capital
    
    if bankroll is not None:
        return kelly_fractional * bankroll
    
    return kelly_fractional


def compute_kelly_fraction(prob_est, cote, fraction=0.25, bankroll=None):
    """Backward-compatible alias used by the UI and orchestrator modules."""
    return calculate_kelly_fraction(
        prob_est=prob_est,
        cote=cote,
        fraction=fraction,
        bankroll=bankroll,
    )


def calculate_kelly_advanced(prob_est, cote, bankroll, win_rate_historique=None,
                               max_stake_pct=0.10, kelly_fraction=0.25):
    """
    Version avancée avec prise en compte de l'historique.
    
    Args:
        prob_est: Probabilité estimée
        cote: Cote décimale
        bankroll: Capital actuel
        win_rate_historique: Taux de réussite historique (optionnel)
        max_stake_pct: Pourcentage maximum du capital à miser
        kelly_fraction: Fraction de Kelly
    
    Returns:
        dict avec mise, fraction, expected_value, etc.
    """
    
    ev = calculate_expected_value(prob_est, cote)
    kelly_fr = calculate_kelly_fraction(prob_est, cote, kelly_fraction)
    mise_kelly = kelly_fr * bankroll
    
    # Ajustement selon l'historique
    ajustement = 1.0
    if win_rate_historique is not None:
        # Si le win rate historique est inférieur à la probabilité estimée,
        # on réduit la mise (le modèle pourrait être trop optimiste)
        if win_rate_historique < prob_est - 0.05:
            ajustement = 0.7
        elif win_rate_historique > prob_est + 0.05:
            ajustement = 1.2  # Légère augmentation si le modèle est prudent
    
    mise_finale = mise_kelly * ajustement
    
    # Appliquer le plafond
    mise_max = bankroll * max_stake_pct
    mise_finale = min(mise_finale, mise_max)
    
    return {
        'prob_est': prob_est,
        'cote': cote,
        'expected_value': ev,
        'kelly_full': kelly_fr / kelly_fraction if kelly_fraction > 0 else 0,
        'kelly_fractional': kelly_fr,
        'mise_conseillee': round(mise_finale, 2),
        'pct_bankroll': mise_finale / bankroll if bankroll > 0 else 0,
        'ajustement': ajustement
    }


# ============================================================
# 2. DÉTECTION DE VALUE BETS
# ============================================================

def find_value_bets(matches_df, model_predict_func, bankroll=1000.0,
                    kelly_fraction=0.25, seuil_ev=0.02, seuil_kelly=0.005):
    """
    Analyse une liste de matchs et détecte les value bets.
    
    Args:
        matches_df: DataFrame avec les matchs et cotes
        model_predict_func: Fonction qui prend un match et retourne la probabilité estimée
        bankroll: Capital total
        kelly_fraction: Fraction de Kelly
        seuil_ev: Seuil minimum d'expected value (ex: 0.02 = 2%)
        seuil_kelly: Seuil minimum de Kelly fractionnel
    
    Returns:
        list: Liste des value bets détectés
    """
    
    bets = []
    
    for idx, row in matches_df.iterrows():
        try:
            # Récupérer la probabilité estimée par le modèle
            prob_home = model_predict_func(row)
            
            # Analyser le pari "victoire domicile"
            cote_home = row.get('cote_home')
            if cote_home and cote_home > 1.0:
                bet = _analyze_single_bet(
                    match=f"{row['home_team']} vs {row['away_team']}",
                    prob_est=prob_home,
                    cote=cote_home,
                    type_paris='home',
                    bankroll=bankroll,
                    kelly_fraction=kelly_fraction,
                    seuil_ev=seuil_ev,
                    seuil_kelly=seuil_kelly
                )
                if bet:
                    bets.append(bet)
            
            # Analyser aussi le pari "victoire extérieur"
            cote_away = row.get('cote_away')
            if cote_away and cote_away > 1.0:
                prob_away = 1 - prob_home  # Simplification
                bet = _analyze_single_bet(
                    match=f"{row['away_team']} @ {row['home_team']}",
                    prob_est=prob_away,
                    cote=cote_away,
                    type_paris='away',
                    bankroll=bankroll,
                    kelly_fraction=kelly_fraction,
                    seuil_ev=seuil_ev,
                    seuil_kelly=seuil_kelly
                )
                if bet:
                    bets.append(bet)
            
            # Analyser le match nul
            cote_draw = row.get('cote_draw')
            if cote_draw and cote_draw > 1.0:
                # Estimation simple de la probabilité de nul
                prob_draw = estimate_draw_probability(prob_home)
                bet = _analyze_single_bet(
                    match=f"Nul : {row['home_team']} vs {row['away_team']}",
                    prob_est=prob_draw,
                    cote=cote_draw,
                    type_paris='draw',
                    bankroll=bankroll,
                    kelly_fraction=kelly_fraction,
                    seuil_ev=seuil_ev,
                    seuil_kelly=seuil_kelly
                )
                if bet:
                    bets.append(bet)
                    
        except Exception as e:
            print(f"⚠️  Erreur analyse match {row.get('home_team', '?')} vs {row.get('away_team', '?')}: {e}")
            continue
    
    # Trier par expected value décroissante
    bets.sort(key=lambda x: x['expected_value'], reverse=True)
    
    return bets


def _get_risk_profile(prob_combined):
    """Retourne un profil de risque lisible basé sur la probabilité combinée."""
    if prob_combined >= 0.25:
        return {
            'label': 'Faible',
            'badge': '🟢',
            'description': 'Combiné prudent, bonne probabilité de gagner',
        }
    if prob_combined >= 0.15:
        return {
            'label': 'Moyen',
            'badge': '🟡',
            'description': 'Combiné équilibré entre rendement et risque',
        }
    return {
        'label': 'Élevé',
        'badge': '🔴',
        'description': 'Combiné agressif, rendement élevé mais plus volatil',
    }


def combine_bets(single_bets, max_matches=3, kelly_fraction=0.10,
                 bankroll=1000.0, seuil_ev=0.0, max_results=3):
    """
    Génère des paris combinés rentables à partir des paris simples détectés.

    Args:
        single_bets: Liste de paris simples (dictionnaires)
        max_matches: Nombre maximum de sélections dans un combiné
        kelly_fraction: Fraction de Kelly dédiée aux combinés (0.10 = 10%)
        bankroll: Capital total pour calculer la mise
        seuil_ev: EV minimum exigé pour conserver un combiné
        max_results: Nombre maximum de combinés à retourner

    Returns:
        list: Liste de combinés triés par EV décroissante
    """

    if not single_bets:
        return []

    max_matches = max(2, int(max_matches))
    prepared = []

    for bet in single_bets:
        prob_est = bet.get('prob_est', bet.get('prob_home'))
        cote = bet.get('cote', bet.get('cote_home'))
        if prob_est is None or cote is None:
            continue
        if not (0 < prob_est < 1) or cote <= 1:
            continue

        prepared.append({
            'sport_name': bet.get('sport_name', '-'),
            'match': bet.get('match', 'Match inconnu'),
            'selection': bet.get('type_paris', 'home'),
            'prob_est': float(prob_est),
            'cote': float(cote),
            'expected_value': float(bet.get('expected_value', calculate_expected_value(prob_est, cote))),
        })

    if len(prepared) < 2:
        return []

    combined = []
    upper = min(max_matches, len(prepared))
    for size in range(2, upper + 1):
        for combo in combinations(prepared, size):
            prob_combined = 1.0
            cote_combined = 1.0
            for selection in combo:
                prob_combined *= selection['prob_est']
                cote_combined *= selection['cote']

            ev_combined = calculate_expected_value(prob_combined, cote_combined)
            if ev_combined <= seuil_ev:
                continue

            kelly_pct = calculate_kelly_fraction(
                prob_combined,
                cote_combined,
                fraction=kelly_fraction,
            )
            if kelly_pct <= 0:
                continue

            stake = bankroll * kelly_pct
            payout = stake * cote_combined
            risk = _get_risk_profile(prob_combined)

            combined.append({
                'nb_matches': size,
                'selections': list(combo),
                'cote_totale': cote_combined,
                'probabilite': prob_combined,
                'expected_value': ev_combined,
                'kelly_fraction': kelly_pct,
                'mise': round(stake, 2),
                'gain_potentiel': round(payout, 2),
                'risk_label': risk['label'],
                'risk_badge': risk['badge'],
                'risk_description': risk['description'],
            })

    combined.sort(key=lambda item: (item['expected_value'], item['probabilite']), reverse=True)
    return combined[:max_results]


def _analyze_single_bet(match, prob_est, cote, type_paris, bankroll,
                         kelly_fraction, seuil_ev, seuil_kelly):
    """
    Analyse un pari individuel.
    
    Returns:
        dict ou None si pas de value
    """
    
    # Calcul de l'espérance
    ev = calculate_expected_value(prob_est, cote)
    
    # Vérifier les seuils
    if ev <= seuil_ev:
        return None
    
    # Calcul de Kelly
    kelly_fr = calculate_kelly_fraction(prob_est, cote, kelly_fraction)
    
    if kelly_fr <= seuil_kelly:
        return None
    
    # Calcul de la mise
    mise = kelly_fr * bankroll
    
    # Probabilité implicite du marché
    prob_implicite = 1 / cote if cote > 0 else 0
    
    return {
        'match': match,
        'type_paris': type_paris,
        'prob_est': prob_est,
        'prob_implicite': prob_implicite,
        'cote': cote,
        'expected_value': ev,
        'kelly_fraction': kelly_fr,
        'mise_conseillee': round(mise, 2),
        'avantage': ev  # Synonyme pour affichage
    }


def estimate_draw_probability(prob_home):
    """
    Estime la probabilité de match nul à partir de la probabilité de victoire domicile.
    Basé sur une relation empirique observée dans les grands championnats.
    
    Args:
        prob_home: Probabilité estimée de victoire domicile
    
    Returns:
        float: Probabilité estimée de match nul
    """
    # Le nul est plus probable quand les équipes sont équilibrées
    # Moins probable quand il y a un grand favori
    
    # Distance par rapport à l'équilibre parfait (0.33)
    distance = abs(prob_home - 0.33)
    
    # Maximum de probabilité de nul autour de l'équilibre
    prob_nul_max = 0.30  # 30% max
    prob_nul_min = 0.10  # 10% min
    
    # Décroissance exponentielle
    prob_nul = prob_nul_min + (prob_nul_max - prob_nul_min) * math.exp(-distance * 3)
    
    # Ajuster pour que prob_home + prob_away + prob_nul ≈ 1
    prob_away = 1 - prob_home - prob_nul
    if prob_away < 0.05:
        prob_away = 0.05
        prob_nul = 1 - prob_home - prob_away
    
    return prob_nul


# ============================================================
# 3. FILTRES DE QUALITÉ
# ============================================================

def filter_quality_bets(bets, min_cote=1.50, max_cote=5.00, 
                         min_ev=0.02, max_mise_pct=0.10):
    """
    Filtre les value bets selon des critères de qualité.
    
    Args:
        bets: Liste des value bets
        min_cote: Cote minimum acceptable
        max_cote: Cote maximum acceptable
        min_ev: Expected value minimum
        max_mise_pct: Pourcentage maximum du capital
    
    Returns:
        list: Bets filtrés
    """
    
    filtered = []
    
    for bet in bets:
        # Filtre sur la cote
        if bet['cote'] < min_cote or bet['cote'] > max_cote:
            continue
        
        # Filtre sur l'EV
        if bet['expected_value'] < min_ev:
            continue
        
        # Filtre sur la mise
        if bet.get('kelly_fraction', 0) > max_mise_pct:
            # Réduire la mise au maximum autorisé
            bet = bet.copy()
            bet['kelly_fraction'] = max_mise_pct
            bet['mise_conseillee'] = bet['mise_conseillee'] * (max_mise_pct / bet.get('kelly_fraction_original', max_mise_pct))
        
        filtered.append(bet)
    
    return filtered


def filter_correlated_bets(bets):
    """
    Détecte et gère les paris corrélés (ex: parier sur les deux équipes du même match).
    
    Args:
        bets: Liste des value bets
    
    Returns:
        list: Bets avec doublons résolus
    """
    
    if len(bets) <= 1:
        return bets
    
    # Regrouper par match
    match_groups = {}
    for bet in bets:
        # Extraire les équipes du nom du match
        match_key = bet['match'].replace('Nul : ', '').replace(' @ ', ' vs ')
        if match_key not in match_groups:
            match_groups[match_key] = []
        match_groups[match_key].append(bet)
    
    resolved = []
    
    for match_key, group in match_groups.items():
        if len(group) == 1:
            resolved.append(group[0])
        else:
            # Plusieurs paris sur le même match : garder le meilleur EV
            best = max(group, key=lambda x: x['expected_value'])
            resolved.append(best)
            # Logguer l'exclusion
            for bet in group:
                if bet != best:
                    print(f"⚠️  Pari exclu (corrélé) : {bet['match']} ({bet['type_paris']}, EV={bet['expected_value']:.1%})")
    
    return resolved


# ============================================================
# 4. GESTION DU BANKROLL
# ============================================================

class BankrollManager:
    """
    Gère le capital de paris avec suivi et règles de prudence.
    """
    
    def __init__(self, initial_bankroll=1000.0):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.history = []
        self.load_state()
    
    def load_state(self):
        """Charge l'état depuis la base de données."""
        try:
            config_bankroll = get_config('bankroll')
            if config_bankroll:
                self.current_bankroll = float(config_bankroll)
            
            # Calculer le profit réalisé
            paris = get_paris_termines()
            if not paris.empty:
                paris['profit'] = paris['gain'] - paris['mise']
                profit_total = paris['profit'].sum()
                # Ajuster le bankroll initial si nécessaire
                self.current_bankroll = self.initial_bankroll + profit_total
                
        except Exception as e:
            print(f"⚠️  Erreur chargement bankroll : {e}")
    
    def save_state(self):
        """Sauvegarde le bankroll actuel."""
        set_config('bankroll', str(round(self.current_bankroll, 2)))
    
    def get_bankroll(self):
        return self.current_bankroll
    
    def update_after_bet(self, mise, gain):
        """Met à jour le bankroll après un pari."""
        self.current_bankroll = self.current_bankroll - mise + gain
        self.save_state()
    
    def get_max_stake(self, max_pct=0.10):
        """Retourne la mise maximale autorisée."""
        return self.current_bankroll * max_pct
    
    def get_stats(self):
        """Retourne les statistiques du bankroll."""
        return {
            'bankroll_initial': self.initial_bankroll,
            'bankroll_actuel': self.current_bankroll,
            'variation': self.current_bankroll - self.initial_bankroll,
            'variation_pct': ((self.current_bankroll / self.initial_bankroll) - 1) * 100
        }


# ============================================================
# 5. ANALYSE DE LA VALUE DÉTECTÉE
# ============================================================

def analyze_value_quality(bet, historique_paris=None):
    """
    Analyse la qualité d'un value bet avec un score composite.
    
    Critères :
    - EV élevé = meilleur score
    - Cote dans une plage raisonnable (1.50-4.00)
    - Cohérence avec l'historique
    
    Args:
        bet: Dictionnaire du value bet
        historique_paris: DataFrame des paris passés (optionnel)
    
    Returns:
        dict avec score_qualite (0-100) et détails
    """
    
    score = 0
    details = []
    
    # 1. Score basé sur l'EV (max 40 points)
    ev = bet.get('expected_value', 0)
    score_ev = min(40, max(0, ev * 200))  # 5% EV = 10 points, 20% EV = 40 points
    score += score_ev
    details.append(f"Score EV : {score_ev:.0f}/40")
    
    # 2. Score basé sur la cote (max 20 points)
    cote = bet.get('cote', 1)
    if 1.50 <= cote <= 3.50:
        score_cote = 20
    elif 3.50 < cote <= 5.00:
        score_cote = 15
    elif 1.20 <= cote < 1.50:
        score_cote = 10
    else:
        score_cote = 5
    score += score_cote
    details.append(f"Score cote : {score_cote}/20")
    
    # 3. Score basé sur l'écart probabilité estimée vs implicite (max 20 points)
    prob_est = bet.get('prob_est', 0)
    prob_imp = bet.get('prob_implicite', 0)
    ecart = prob_est - prob_imp
    score_ecart = min(20, max(0, ecart * 100))
    score += score_ecart
    details.append(f"Score écart : {score_ecart:.0f}/20")
    
    # 4. Score basé sur la cohérence historique (max 20 points)
    score_hist = 10  # Valeur par défaut
    if historique_paris is not None and not historique_paris.empty:
        # Vérifier si on a déjà parié sur des cotes similaires avec succès
        cotes_similaires = historique_paris[
            (historique_paris['cote'] >= cote * 0.8) & 
            (historique_paris['cote'] <= cote * 1.2)
        ]
        if len(cotes_similaires) > 0:
            win_rate_similaire = len(cotes_similaires[cotes_similaires['resultat'] == 'Win']) / len(cotes_similaires)
            if win_rate_similaire > 0.5:
                score_hist = 20
            elif win_rate_similaire > 0.4:
                score_hist = 15
            else:
                score_hist = 8
    score += score_hist
    details.append(f"Score historique : {score_hist}/20")
    
    # Score total
    score_total = min(100, score)
    
    # Interprétation
    if score_total >= 80:
        interpretation = "⭐⭐⭐ Excellent value bet"
    elif score_total >= 60:
        interpretation = "⭐⭐ Bon value bet"
    elif score_total >= 40:
        interpretation = "⭐ Value bet acceptable"
    else:
        interpretation = "⚠️ Value bet discutable"
    
    return {
        'score_qualite': round(score_total, 0),
        'interpretation': interpretation,
        'details': details
    }


# ============================================================
# 6. RAPPORT DE VALUE BETS
# ============================================================

def generate_value_report(bets, bankroll):
    """
    Génère un rapport formaté des value bets détectés.
    
    Args:
        bets: Liste des value bets
        bankroll: Capital actuel
    
    Returns:
        str: Rapport formaté
    """
    
    if not bets:
        return "❌ Aucun value bet détecté."
    
    lignes = []
    lignes.append("\n" + "=" * 70)
    lignes.append("  📊 RAPPORT DE VALUE BETS")
    lignes.append("=" * 70)
    lignes.append(f"  Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    lignes.append(f"  Bankroll : {bankroll:.2f}€")
    lignes.append(f"  Nombre d'opportunités : {len(bets)}")
    lignes.append("-" * 70)
    
    mise_totale = 0
    
    for i, bet in enumerate(bets, 1):
        lignes.append(f"\n  {i}. {bet['match']}")
        lignes.append(f"     Type de pari : {bet['type_paris']}")
        lignes.append(f"     Probabilité estimée : {bet['prob_est']:.1%}")
        lignes.append(f"     Probabilité implicite : {bet['prob_implicite']:.1%}")
        lignes.append(f"     Cote : {bet['cote']:.2f}")
        lignes.append(f"     Expected Value : {bet['expected_value']:.1%}")
        lignes.append(f"     Kelly fraction : {bet['kelly_fraction']:.2%}")
        lignes.append(f"     Mise conseillée : {bet['mise_conseillee']:.2f}€")
        
        mise_totale += bet['mise_conseillee']
    
    lignes.append("\n" + "-" * 70)
    lignes.append(f"  Mise totale conseillée : {mise_totale:.2f}€")
    lignes.append(f"  Exposition : {mise_totale/bankroll*100:.1f}% du bankroll")
    lignes.append("=" * 70 + "\n")
    
    return "\n".join(lignes)


# ============================================================
# 7. SIMULATION DE MONTE CARLO
# ============================================================

def monte_carlo_simulation(bets, bankroll, n_simulations=10000):
    """
    Simule les résultats possibles des value bets par Monte Carlo.
    
    Args:
        bets: Liste des value bets
        bankroll: Capital de départ
        n_simulations: Nombre de simulations
    
    Returns:
        dict avec statistiques des résultats simulés
    """
    
    final_bankrolls = []
    
    for _ in range(n_simulations):
        bankroll_sim = bankroll
        
        for bet in bets:
            mise = bet['mise_conseillee']
            prob = bet['prob_est']
            cote = bet['cote']
            
            # Simulation du résultat
            if np.random.random() < prob:
                # Gagné
                bankroll_sim += mise * (cote - 1)
            else:
                # Perdu
                bankroll_sim -= mise
        
        final_bankrolls.append(bankroll_sim)
    
    final_bankrolls = np.array(final_bankrolls)
    
    return {
        'bankroll_initial': bankroll,
        'bankroll_moyen': np.mean(final_bankrolls),
        'bankroll_median': np.median(final_bankrolls),
        'bankroll_min': np.min(final_bankrolls),
        'bankroll_max': np.max(final_bankrolls),
        'ecart_type': np.std(final_bankrolls),
        'probabilite_gain': np.mean(final_bankrolls > bankroll),
        'probabilite_perte': np.mean(final_bankrolls < bankroll),
        'gain_moyen': np.mean(final_bankrolls) - bankroll,
        'value_at_risk_95': bankroll - np.percentile(final_bankrolls, 5),
        'n_simulations': n_simulations
    }


# ============================================================
# 8. TEST
# ============================================================

if __name__ == "__main__":
    print("🧪 Test du module value_detector\n")
    
    # Test 1 : Calcul de base
    print("=" * 60)
    print("TEST 1 : Calcul Kelly de base")
    print("=" * 60)
    
    prob = 0.55
    cote = 2.10
    ev = calculate_expected_value(prob, cote)
    kelly = calculate_kelly_fraction(prob, cote, fraction=0.25)
    
    print(f"   Probabilité estimée : {prob:.1%}")
    print(f"   Cote : {cote:.2f}")
    print(f"   Probabilité implicite : {1/cote:.1%}")
    print(f"   Expected Value : {ev:.1%}")
    print(f"   Kelly complet : {kelly/0.25:.1%}")
    print(f"   Kelly fractionnel (25%) : {kelly:.2%}")
    print(f"   Mise pour 1000€ : {kelly * 1000:.2f}€")
    
    # Test 2 : Tableau de comparaison
    print("\n" + "=" * 60)
    print("TEST 2 : Comparaison probabilité vs cote")
    print("=" * 60)
    
    print(f"\n   {'Prob est':>10} {'Cote':>8} {'EV':>8} {'Kelly 25%':>10} {'Mise/1000€':>12}")
    print("   " + "-" * 50)
    
    for prob in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        for cote in [1.80, 2.00, 2.50, 3.00]:
            ev = calculate_expected_value(prob, cote)
            kelly = calculate_kelly_fraction(prob, cote, 0.25)
            if ev > 0 and kelly > 0.005:
                mise = kelly * 1000
                print(f"   {prob:>10.0%} {cote:>8.2f} {ev:>8.1%} {kelly:>10.2%} {mise:>12.2f}€")
    
    # Test 3 : Kelly avancé
    print("\n" + "=" * 60)
    print("TEST 3 : Kelly avancé avec historique")
    print("=" * 60)
    
    result = calculate_kelly_advanced(
        prob_est=0.55,
        cote=2.10,
        bankroll=1000,
        win_rate_historique=0.48,
        max_stake_pct=0.10,
        kelly_fraction=0.25
    )
    for k, v in result.items():
        print(f"   {k}: {v}")
    
    # Test 4 : Détection de value bets (simulée)
    print("\n" + "=" * 60)
    print("TEST 4 : Détection de value bets")
    print("=" * 60)
    
    # Créer des matchs simulés
    matches_simules = pd.DataFrame([
        {'home_team': 'Arsenal', 'away_team': 'Chelsea', 
         'cote_home': 2.10, 'cote_draw': 3.50, 'cote_away': 3.40},
        {'home_team': 'Liverpool', 'away_team': 'Man United', 
         'cote_home': 1.85, 'cote_draw': 3.80, 'cote_away': 4.10},
        {'home_team': 'Brighton', 'away_team': 'West Ham', 
         'cote_home': 2.50, 'cote_draw': 3.30, 'cote_away': 2.80},
    ])
    
    def mock_predict(row):
        # Simulation : retourne une probabilité basée sur la cote
        if row['home_team'] == 'Arsenal':
            return 0.55  # Value bet (2.10 * 0.55 = 1.155)
        elif row['home_team'] == 'Liverpool':
            return 0.52  # Pas de value (1.85 * 0.52 = 0.962)
        else:
            return 0.45  # Value sur away
    
    bets = find_value_bets(matches_simules, mock_predict, bankroll=1000)
    
    if bets:
        for bet in bets:
            print(f"\n   Match : {bet['match']}")
            print(f"   Type : {bet['type_paris']}")
            print(f"   EV : {bet['expected_value']:.1%}")
            print(f"   Mise : {bet['mise_conseillee']:.2f}€")
    else:
        print("   Aucun value bet détecté.")
    
    # Test 5 : Filtres de qualité
    print("\n" + "=" * 60)
    print("TEST 5 : Filtres de qualité")
    print("=" * 60)
    
    if bets:
        for bet in bets:
            qualite = analyze_value_quality(bet)
            print(f"\n   {bet['match']}")
            print(f"   Score qualité : {qualite['score_qualite']:.0f}/100")
            print(f"   Interprétation : {qualite['interpretation']}")
    
    # Test 6 : Simulation Monte Carlo
    print("\n" + "=" * 60)
    print("TEST 6 : Simulation Monte Carlo")
    print("=" * 60)
    
    if bets:
        sim_result = monte_carlo_simulation(bets, 1000, n_simulations=5000)
        print(f"   Bankroll moyen après paris : {sim_result['bankroll_moyen']:.2f}€")
        print(f"   Probabilité de gain : {sim_result['probabilite_gain']:.1%}")
        print(f"   Gain moyen : {sim_result['gain_moyen']:.2f}€")
        print(f"   Value at Risk (95%) : {sim_result['value_at_risk_95']:.2f}€")
    
    # Test 7 : Rapport
    print("\n" + "=" * 60)
    print("TEST 7 : Génération de rapport")
    print("=" * 60)
    
    if bets:
        rapport = generate_value_report(bets, 1000)
        print(rapport)
    
    print("\n✅ Tests terminés !")