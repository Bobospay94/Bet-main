#!/usr/bin/env python3
"""
Système de Pronostics Value Bets - Orchestrateur Principal
===========================================================

Planifie et exécute automatiquement :
1. L'analyse quotidienne des matchs et la détection de value bets
2. La mise à jour des résultats des matchs terminés
3. Le recalibrage automatique du modèle après chaque nouveau résultat
4. L'envoi des pronostics par email/SMS (optionnel)
5. La génération d'un rapport quotidien

Peut fonctionner :
- En mode planifié (avec schedule) pour un serveur 24/7
- En one-shot pour une exécution manuelle
- En mode démon avec systemd
"""

import schedule
import time
import sys
import os
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports des modules du projet
from db import (
    init_db, get_coefficients, get_matchs_sans_resultat,
    mettre_a_jour_resultat, enregistrer_paris, get_statistiques_paris,
    get_config, ajouter_match_historique, get_matchs_avec_resultat
)
from data_collector import get_live_odds, get_team_stats, update_results
from matrix_builder import build_payoff_matrix
from nash_equilibrium import solve_nash_2x2
from value_detector import compute_kelly_fraction, find_value_bets
from recalibrator import recalibrate

# ============================================================
# CONFIGURATION
# ============================================================

# Fichier de log
LOG_FILE = "systeme_pronos.log"

# Email (optionnel) - à configurer dans config.py
try:
    from config import (
        EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO,
        SMTP_SERVER, SMTP_PORT
    )
    EMAIL_ENABLED = True
except ImportError:
    EMAIL_ENABLED = False
    EMAIL_FROM = None

# Championnat par défaut
SPORT_PAR_DEFAUT = "soccer_epl"

# Heures d'exécution
HEURE_ANALYSE = "08:00"    # Analyse quotidienne des matchs
HEURE_RAPPORT = "20:00"    # Rapport du soir
HEURE_RESULTATS = "23:00"  # Mise à jour des résultats


# ============================================================
# FONCTIONS DE LOG
# ============================================================

def log(message, niveau="INFO"):
    """Écrit un message dans le fichier de log avec horodatage."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ligne = f"[{timestamp}] [{niveau}] {message}"
    print(ligne)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(ligne + "\n")
    except:
        pass


def log_separateur():
    """Ajoute une ligne de séparation dans les logs."""
    log("=" * 70)


# ============================================================
# ANALYSE QUOTIDIENNE
# ============================================================

def analyser_matchs_du_jour(sport=None, envoyer_email=True):
    """
    Fonction principale d'analyse quotidienne.
    
    1. Récupère les cotes du jour
    2. Pour chaque match, construit la matrice de gains
    3. Calcule l'équilibre de Nash et la probabilité fondamentale
    4. Détecte les value bets avec le critère de Kelly
    5. Sauvegarde les opportunités en base
    6. Envoie un email avec les pronostics si configuré
    
    Args:
        sport: Identifiant du championnat (ex: soccer_epl)
        envoyer_email: Si True, envoie les résultats par email
    
    Returns:
        list: Liste des value bets détectés
    """
    if sport is None:
        sport = get_config('championnat_par_defaut') or SPORT_PAR_DEFAUT
    
    log_separateur()
    log(f"🔍 DÉBUT ANALYSE QUOTIDIENNE - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    log(f"   Championnat : {sport}")
    
    bets = []
    
    try:
        # 1. Récupération des cotes
        log("📡 Récupération des cotes...")
        matches = get_live_odds(sport=sport)
        
        if matches.empty:
            log("   Aucun match trouvé pour aujourd'hui.", "WARNING")
            return bets
        
        log(f"   {len(matches)} matchs trouvés.")
        
        # 2. Récupération des coefficients du modèle
        coeffs = get_coefficients()
        log(f"   Coefficients : ailes/pressing={coeffs['bonus_ailes_pressing']:.2f}, "
            f"ailes/bloc={coeffs['bonus_ailes_bloc']:.2f}, "
            f"axe/pressing={coeffs['bonus_axe_pressing']:.2f}, "
            f"axe/bloc={coeffs['bonus_axe_bloc']:.2f}")
        
        # 3. Analyse match par match
        for idx, row in matches.iterrows():
            try:
                home = row['home_team']
                away = row['away_team']
                
                log(f"\n   ⚽ Analyse : {home} vs {away}")
                
                # Récupération des statistiques
                stats_home = get_team_stats(home)
                stats_away = get_team_stats(away)
                
                log(f"      Stats {home}: xG={stats_home['xG_for']}, xGA={stats_home['xG_against']}, "
                    f"Poss={stats_home['possession']}%")
                log(f"      Stats {away}: xG={stats_away['xG_for']}, xGA={stats_away['xG_against']}, "
                    f"Poss={stats_away['possession']}%")
                
                # Stratégies possibles
                strategies_A = ["ailes", "axe"]
                strategies_B = ["pressing_haut", "bloc_bas"]
                
                # Construction de la matrice de gains
                payoff = build_payoff_matrix(
                    stats_home, stats_away,
                    strategies_A, strategies_B,
                    coeffs
                )
                
                log(f"      Matrice gains: "
                    f"[{payoff[0,0]:.2%}, {payoff[0,1]:.2%}; "
                    f"{payoff[1,0]:.2%}, {payoff[1,1]:.2%}]")
                
                # Résolution de l'équilibre de Nash
                p_A, q_B, prob_home = solve_nash_2x2(payoff)
                
                log(f"      Équilibre Nash: p(A1)={p_A:.1%}, q(B1)={q_B:.1%}")
                log(f"      Probabilité estimée victoire {home}: {prob_home:.1%}")
                
                # Cotes
                cote_home = row.get('cote_home')
                cote_draw = row.get('cote_draw')
                cote_away = row.get('cote_away')
                
                if cote_home:
                    prob_implicite = 1 / cote_home
                    log(f"      Cote {home}: {cote_home:.2f} (prob implicite: {prob_implicite:.1%})")
                
                # Sauvegarde dans l'historique
                match_id = ajouter_match_historique(
                    home_team=home,
                    away_team=away,
                    date_match=row.get('match_date', datetime.now().isoformat()),
                    xg_home=stats_home['xG_for'],
                    xg_away=stats_away['xG_for'],
                    possession_home=stats_home['possession'],
                    possession_away=stats_away['possession'],
                    strategies={"A": "ailes", "B": "pressing_haut"},  # Stratégie dominante
                    prob_estimee_home=prob_home,
                    cote_home=cote_home,
                    cote_draw=cote_draw,
                    cote_away=cote_away
                )
                
                # 4. Détection de value bet
                if cote_home and cote_home > 1.0:
                    kelly_fraction = float(get_config('kelly_fraction') or 0.25)
                    seuil_value = float(get_config('seuil_value') or 0.02)
                    bankroll = float(get_config('bankroll') or 1000.0)
                    
                    expected_value = (prob_home * cote_home) - 1
                    kelly = compute_kelly_fraction(prob_home, cote_home, kelly_fraction)
                    mise_conseillee = bankroll * kelly
                    
                    if expected_value > seuil_value and kelly > 0.005:
                        log(f"      ✅ VALUE BET DÉTECTÉ ! EV={expected_value:.1%}, "
                            f"Kelly={kelly:.1%}, Mise={mise_conseillee:.2f}€")
                        
                        bet = {
                            'match': f"{home} vs {away}",
                            'home_team': home,
                            'away_team': away,
                            'prob_est': prob_home,
                            'cote': cote_home,
                            'prob_implicite': prob_implicite,
                            'expected_value': expected_value,
                            'kelly_stake': kelly,
                            'mise_conseillee': mise_conseillee,
                            'match_id': match_id,
                            'strategie_A': f"Ailes: {p_A:.0%} / Axe: {1-p_A:.0%}",
                            'strategie_B': f"Pressing: {q_B:.0%} / Bloc: {1-q_B:.0%}"
                        }
                        bets.append(bet)
                    else:
                        log(f"      ❌ Pas de value bet. EV={expected_value:.1%} "
                            f"(seuil: {seuil_value:.1%})")
                else:
                    log(f"      ⚠️  Cote non disponible pour {home}")
                    
            except Exception as e:
                log(f"      ❌ Erreur lors de l'analyse de {home} vs {away}: {e}", "ERROR")
                log(traceback.format_exc(), "ERROR")
                continue
        
        # 5. Résumé
        log(f"\n📊 RÉSUMÉ : {len(bets)} value bet(s) détecté(s) sur {len(matches)} matchs.")
        
        if bets:
            log("\n   Opportunités :")
            for i, bet in enumerate(bets, 1):
                log(f"   {i}. {bet['match']} - Prob: {bet['prob_est']:.1%} - "
                    f"Cote: {bet['cote']:.2f} - EV: {bet['expected_value']:.1%} - "
                    f"Mise: {bet['mise_conseillee']:.2f}€")
        
        # 6. Envoi par email
        if envoyer_email and EMAIL_ENABLED and bets:
            envoyer_email_pronostics(bets, sport)
        
        log("✅ ANALYSE TERMINÉE")
        
    except Exception as e:
        log(f"❌ ERREUR GLOBALE lors de l'analyse : {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
    
    return bets


# ============================================================
# MISE À JOUR DES RÉSULTATS
# ============================================================

def mettre_a_jour_resultats():
    """
    Vérifie les matchs en attente et met à jour leurs résultats.
    Lance le recalibrage si de nouveaux résultats sont disponibles.
    """
    log_separateur()
    log(f"🔄 MISE À JOUR DES RÉSULTATS - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    try:
        # Mettre à jour les résultats via l'API
        updated = update_results()
        
        if updated:
            log("   Nouveaux résultats détectés. Lancement du recalibrage...")
            recalibrate()
        else:
            log("   Aucun nouveau résultat à traiter.")
        
        # Vérifier aussi les paris en attente
        from db import get_paris_en_attente, mettre_a_jour_paris
        
        paris_en_attente = get_paris_en_attente()
        if not paris_en_attente.empty:
            log(f"   {len(paris_en_attente)} pari(s) en attente de résultat.")
            
            for _, pari in paris_en_attente.iterrows():
                # Vérifier si le match associé a un résultat
                if pari['match_id']:
                    from db import get_matchs_avec_resultat
                    matchs_resultats = get_matchs_avec_resultat()
                    
                    match_trouve = matchs_resultats[matchs_resultats['id'] == pari['match_id']]
                    if not match_trouve.empty:
                        match = match_trouve.iloc[0]
                        if match['resultat'] == 'home':
                            # Pari gagné si on avait misé sur home
                            if pari['type_paris'] == 'home':
                                gain = pari['mise'] * pari['cote']
                                mettre_a_jour_paris(pari['id'], 'Win', gain)
                                log(f"   ✅ Pari gagné : {pari['match_nom']} - +{gain:.2f}€")
                            else:
                                mettre_a_jour_paris(pari['id'], 'Loss', 0)
                                log(f"   ❌ Pari perdu : {pari['match_nom']}")
                        elif match['resultat'] == 'draw':
                            if pari['type_paris'] == 'draw':
                                gain = pari['mise'] * pari['cote']
                                mettre_a_jour_paris(pari['id'], 'Win', gain)
                            else:
                                mettre_a_jour_paris(pari['id'], 'Loss', 0)
                        elif match['resultat'] == 'away':
                            if pari['type_paris'] == 'away':
                                gain = pari['mise'] * pari['cote']
                                mettre_a_jour_paris(pari['id'], 'Win', gain)
                            else:
                                mettre_a_jour_paris(pari['id'], 'Loss', 0)
        
        log("✅ MISE À JOUR TERMINÉE")
        
    except Exception as e:
        log(f"❌ Erreur lors de la mise à jour des résultats : {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")


# ============================================================
# RAPPORT QUOTIDIEN
# ============================================================

def generer_rapport_quotidien(envoyer_email=True):
    """
    Génère un rapport de performance quotidien.
    """
    log_separateur()
    log(f"📊 RAPPORT QUOTIDIEN - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    try:
        stats = get_statistiques_paris()
        
        rapport = f"""
╔══════════════════════════════════════════════════════╗
║         RAPPORT QUOTIDIEN - PRONOSTICS              ║
╚══════════════════════════════════════════════════════╝

📅 Date : {datetime.now().strftime('%d/%m/%Y')}

📈 STATISTIQUES GLOBALES :
   • Nombre total de paris : {stats['nb_paris']}
   • Paris gagnés : {stats['nb_wins']}
   • Paris perdus : {stats['nb_losses']}
   • Taux de réussite : {stats['win_rate']}%
   • Cote moyenne gagnante : {stats['cote_moyenne']}

💰 PERFORMANCE FINANCIÈRE :
   • Mise totale : {stats['mise_totale']:.2f}€
   • Gain total : {stats['gain_total']:.2f}€
   • Profit net : {stats['profit_total']:.2f}€
   • ROI : {stats['roi']}%

🔧 MODÈLE :
   • Coefficients : {get_coefficients()}
   • Matchs en historique : {get_config('nb_matchs_historique') or 'N/A'}

══════════════════════════════════════════════════════
        """
        
        log(rapport)
        
        # Envoyer par email
        if envoyer_email and EMAIL_ENABLED:
            envoyer_email_rapport(rapport, stats)
        
        return rapport
        
    except Exception as e:
        log(f"❌ Erreur génération rapport : {e}", "ERROR")


# ============================================================
# ENVOI D'EMAILS
# ============================================================

def envoyer_email_pronostics(bets, sport):
    """Envoie les value bets détectés par email."""
    if not EMAIL_ENABLED:
        return
    
    try:
        sujet = f"⚽ Pronostics Value Bets - {datetime.now().strftime('%d/%m/%Y')}"
        
        corps = f"""
        <h2>🔍 Value Bets détectés - {datetime.now().strftime('%d/%m/%Y')}</h2>
        <p>Championnat : <b>{sport}</b></p>
        <p><b>{len(bets)} opportunité(s) trouvée(s)</b></p>
        <hr>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse; width:100%">
            <tr style="background-color:#4CAF50; color:white">
                <th>Match</th>
                <th>Prob. estimée</th>
                <th>Cote</th>
                <th>EV</th>
                <th>Mise conseillée</th>
                <th>Stratégie</th>
            </tr>
        """
        
        for bet in bets:
            corps += f"""
            <tr>
                <td><b>{bet['match']}</b></td>
                <td>{bet['prob_est']:.1%}</td>
                <td>{bet['cote']:.2f}</td>
                <td style="color:{'green' if bet['expected_value'] > 0 else 'red'}">
                    {bet['expected_value']:.1%}
                </td>
                <td>{bet['mise_conseillee']:.2f}€</td>
                <td>{bet['strategie_A']}</td>
            </tr>
            """
        
        corps += """
        </table>
        <hr>
        <p><small>⚠️ Ces pronostics sont basés sur un modèle mathématique. 
        Les paris sportifs comportent des risques. Ne misez que ce que vous pouvez perdre.</small></p>
        """
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = sujet
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg.attach(MIMEText(corps, "html", "utf-8"))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        
        log(f"📧 Email de pronostics envoyé à {EMAIL_TO}")
        
    except Exception as e:
        log(f"❌ Erreur envoi email pronostics : {e}", "ERROR")


def envoyer_email_rapport(rapport, stats):
    """Envoie le rapport quotidien par email."""
    if not EMAIL_ENABLED:
        return
    
    try:
        sujet = f"📊 Rapport Pronostics - {datetime.now().strftime('%d/%m/%Y')}"
        
        msg = MIMEText(rapport, "plain", "utf-8")
        msg["Subject"] = sujet
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        
        log(f"📧 Rapport quotidien envoyé à {EMAIL_TO}")
        
    except Exception as e:
        log(f"❌ Erreur envoi rapport : {e}", "ERROR")


# ============================================================
# MODE ONE-SHOT (exécution manuelle)
# ============================================================

def run_once(sport=None):
    """
    Exécute une analyse complète une seule fois.
    Utile pour tester ou lancer manuellement.
    """
    log("🚀 EXÉCUTION ONE-SHOT")
    
    # Analyse des matchs
    bets = analyser_matchs_du_jour(sport=sport, envoyer_email=False)
    
    # Affichage console
    if bets:
        print("\n" + "=" * 60)
        print("✅ VALUE BETS DÉTECTÉS :")
        print("=" * 60)
        for i, bet in enumerate(bets, 1):
            print(f"\n{i}. {bet['match']}")
            print(f"   Probabilité estimée : {bet['prob_est']:.1%}")
            print(f"   Cote bookmaker     : {bet['cote']:.2f}")
            print(f"   Valeur attendue    : {bet['expected_value']:.1%}")
            print(f"   Mise conseillée    : {bet['mise_conseillee']:.2f}€")
            print(f"   Stratégie optimale : {bet['strategie_A']}")
        print("\n" + "=" * 60)
    else:
        print("\n❌ Aucun value bet détecté aujourd'hui.")
    
    return bets


# ============================================================
# PLANIFICATEUR (mode serveur 24/7)
# ============================================================

def start_scheduler():
    """
    Démarre le planificateur de tâches.
    Analyse chaque jour à 08:00, rapport à 20:00, résultats toutes les heures.
    """
    log("=" * 70)
    log("🤖 SYSTÈME DE PRONOSTICS AUTOMATIQUES - DÉMARRAGE")
    log(f"   Heure démarrage : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    log(f"   Tâches planifiées :")
    log(f"     - Analyse quotidienne : {HEURE_ANALYSE}")
    log(f"     - Rapport quotidien   : {HEURE_RAPPORT}")
    log(f"     - Mise à jour résultats : toutes les heures")
    log("=" * 70)
    
    # Planification des tâches
    schedule.every().day.at(HEURE_ANALYSE).do(analyser_matchs_du_jour)
    schedule.every().day.at(HEURE_RAPPORT).do(generer_rapport_quotidien)
    schedule.every().hour.do(mettre_a_jour_resultats)
    
    log("✅ Planificateur démarré. En attente des tâches...")
    
    # Boucle principale
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Vérifie toutes les minutes
            
    except KeyboardInterrupt:
        log("\n⏹️  Arrêt demandé par l'utilisateur.")
        log("👋 Système arrêté.")
    except Exception as e:
        log(f"❌ Erreur fatale : {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")


# ============================================================
# POINT D'ENTRÉE
# ============================================================

def print_banner():
    """Affiche la bannière de démarrage."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║   ⚽  SYSTÈME DE PRONOSTICS VALUE BETS  ⚽            ║
    ║   Théorie des Jeux & Équilibre de Nash                ║
    ║   Recalibrage automatique par régression logistique   ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Point d'entrée principal."""
    print_banner()
    
    # Initialisation de la base de données
    print("📂 Initialisation de la base de données...")
    init_db()
    print("✅ Base de données prête.\n")
    
    # Déterminer le mode de fonctionnement
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "scheduler"
    
    if mode == "once":
        # Mode one-shot : analyse et affichage
        sport = sys.argv[2] if len(sys.argv) > 2 else None
        run_once(sport=sport)
        
    elif mode == "scheduler":
        # Mode planifié : tourne en continu
        start_scheduler()
        
    elif mode == "update":
        # Mode mise à jour uniquement
        mettre_a_jour_resultats()
        
    elif mode == "report":
        # Mode rapport uniquement
        generer_rapport_quotidien(envoyer_email=False)
        
    elif mode == "recalibrate":
        # Mode recalibrage manuel
        log("🔧 Recalibrage manuel lancé...")
        recalibrate()
        
    elif mode == "test":
        # Mode test
        log("🧪 Mode test : analyse sans email")
        analyser_matchs_du_jour(envoyer_email=False)
        mettre_a_jour_resultats()
        generer_rapport_quotidien(envoyer_email=False)
        
    else:
        print(f"""
Usage: python main.py [mode] [options]

Modes disponibles:
  once [sport]   : Analyse unique et affichage dans le terminal
  scheduler      : Démarre le planificateur automatique (défaut)
  update         : Mise à jour des résultats uniquement
  report         : Génère le rapport quotidien
  recalibrate    : Recalibrage manuel du modèle
  test           : Mode test complet (analyse + màj + rapport)

Exemples:
  python main.py once
  python main.py once soccer_france_ligue_one
  python main.py scheduler
  python main.py update
        """)

if __name__ == "__main__":
    main()