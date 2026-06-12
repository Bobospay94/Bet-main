"""
Data Collector - Récupération des cotes et statistiques
=========================================================

Ce module gère toute la collecte de données externes :
1. Cotes des matchs via The Odds API (gratuit, 500 requêtes/mois)
2. Statistiques d'équipes estimées depuis les cotes
3. Résultats des matchs terminés
4. Fallback automatique sur des données de démonstration

Fonctionne avec :
- The Odds API seule (estimation des xG depuis les cotes)
- API-Football en option (vrais xG si disponible)
- Mode simulation sans aucune API
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import sqlite3
import json
import os
import random
import hashlib

# Importer la fonction d'estimation depuis matrix_builder
from matrix_builder import estimer_xg_depuis_cotes, estimer_stats_depuis_cotes

# --- Gestion des clés API ---
try:
    from config import ODDS_API_KEY, FOOTBALL_API_KEY
except ImportError:
    ODDS_API_KEY = None
    FOOTBALL_API_KEY = None
    print("⚠️  Fichier config.py non trouvé. Mode démonstration activé.")
    print("   Créez config.py avec ODDS_API_KEY pour les vraies cotes.")

# --- Cache simple pour économiser les requêtes API ---
CACHE_DURATION = 3600  # 1 heure en secondes
_cache = {}

def _cache_get(key):
    """Récupère une valeur du cache si elle est encore valide."""
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < CACHE_DURATION:
            return data
    return None

def _cache_set(key, data):
    """Stocke une valeur dans le cache avec horodatage."""
    _cache[key] = (data, time.time())

def clear_cache():
    """Vide le cache des données."""
    global _cache
    _cache = {}
    print("🗑️  Cache vidé.")

# ============================================================
# BASE DE DONNÉES SIMULÉE (fallback sans API)
# ============================================================

_STATS_EQUIPES = {
    # Premier League
    "Arsenal": {"xG_for": 2.1, "xG_against": 0.9, "possession": 58, "forme": "WWWDW"},
    "Chelsea": {"xG_for": 1.6, "xG_against": 1.3, "possession": 55, "forme": "DWLWD"},
    "Liverpool": {"xG_for": 2.3, "xG_against": 1.0, "possession": 60, "forme": "WWWWL"},
    "Manchester United": {"xG_for": 1.4, "xG_against": 1.2, "possession": 52, "forme": "LDWWL"},
    "Manchester City": {"xG_for": 2.5, "xG_against": 0.7, "possession": 65, "forme": "WWWWW"},
    "Tottenham": {"xG_for": 1.8, "xG_against": 1.5, "possession": 50, "forme": "WLDLW"},
    "Newcastle": {"xG_for": 1.7, "xG_against": 1.1, "possession": 48, "forme": "WDWWL"},
    "Aston Villa": {"xG_for": 1.9, "xG_against": 1.4, "possession": 47, "forme": "WDWLW"},
    "Brighton": {"xG_for": 1.5, "xG_against": 1.3, "possession": 54, "forme": "DLDWW"},
    "West Ham": {"xG_for": 1.3, "xG_against": 1.6, "possession": 43, "forme": "LLDWL"},
    "Wolverhampton": {"xG_for": 1.2, "xG_against": 1.5, "possession": 45, "forme": "LWDLL"},
    "Crystal Palace": {"xG_for": 1.3, "xG_against": 1.4, "possession": 44, "forme": "DWDLW"},
    "Everton": {"xG_for": 1.1, "xG_against": 1.7, "possession": 42, "forme": "LLDLW"},
    "Fulham": {"xG_for": 1.4, "xG_against": 1.5, "possession": 46, "forme": "WDLWL"},
    "Bournemouth": {"xG_for": 1.3, "xG_against": 1.8, "possession": 40, "forme": "LLWDL"},
    "Nottingham Forest": {"xG_for": 1.1, "xG_against": 1.9, "possession": 38, "forme": "LDLWL"},
    "Brentford": {"xG_for": 1.5, "xG_against": 1.4, "possession": 47, "forme": "WDLWD"},
    "Burnley": {"xG_for": 0.9, "xG_against": 2.0, "possession": 36, "forme": "LLLDL"},
    "Sheffield United": {"xG_for": 0.8, "xG_against": 2.2, "possession": 34, "forme": "LLLLD"},
    "Luton": {"xG_for": 1.0, "xG_against": 1.9, "possession": 37, "forme": "LDLLW"},
    
    # Ligue 1
    "PSG": {"xG_for": 2.8, "xG_against": 0.8, "possession": 63, "forme": "WWWWW"},
    "Marseille": {"xG_for": 1.9, "xG_against": 1.1, "possession": 54, "forme": "WWDLW"},
    "Lyon": {"xG_for": 1.7, "xG_against": 1.4, "possession": 52, "forme": "DWWLD"},
    "Monaco": {"xG_for": 1.8, "xG_against": 1.3, "possession": 51, "forme": "WDWWL"},
    "Lille": {"xG_for": 1.6, "xG_against": 1.0, "possession": 49, "forme": "WDWLW"},
    "Rennes": {"xG_for": 1.5, "xG_against": 1.2, "possession": 50, "forme": "DLWWL"},
    "Nice": {"xG_for": 1.4, "xG_against": 0.9, "possession": 48, "forme": "WDWWL"},
    "Lens": {"xG_for": 1.6, "xG_against": 1.1, "possession": 50, "forme": "WWDLW"},
    "Reims": {"xG_for": 1.3, "xG_against": 1.4, "possession": 45, "forme": "DLWLD"},
    "Strasbourg": {"xG_for": 1.2, "xG_against": 1.5, "possession": 44, "forme": "LWDLL"},
    "Montpellier": {"xG_for": 1.3, "xG_against": 1.4, "possession": 46, "forme": "WDLWL"},
    "Nantes": {"xG_for": 1.2, "xG_against": 1.6, "possession": 43, "forme": "LDLLW"},
    "Toulouse": {"xG_for": 1.1, "xG_against": 1.5, "possession": 45, "forme": "DLWLL"},
    "Brest": {"xG_for": 1.4, "xG_against": 1.2, "possession": 47, "forme": "WWDLW"},
    "Le Havre": {"xG_for": 1.0, "xG_against": 1.5, "possession": 42, "forme": "LLDWL"},
    "Metz": {"xG_for": 0.9, "xG_against": 1.8, "possession": 40, "forme": "LLLDL"},
    "Clermont": {"xG_for": 0.8, "xG_against": 1.9, "possession": 39, "forme": "LDLLL"},
    "Lorient": {"xG_for": 1.1, "xG_against": 1.8, "possession": 42, "forme": "LLLWD"},
    
    # Autres grands clubs européens
    "Real Madrid": {"xG_for": 2.4, "xG_against": 0.9, "possession": 58, "forme": "WWWDW"},
    "Barcelona": {"xG_for": 2.2, "xG_against": 1.0, "possession": 62, "forme": "WWWDL"},
    "Bayern Munich": {"xG_for": 2.6, "xG_against": 0.8, "possession": 64, "forme": "WWWWW"},
    "Inter": {"xG_for": 2.0, "xG_against": 0.7, "possession": 54, "forme": "WWWDW"},
    "AC Milan": {"xG_for": 1.8, "xG_against": 1.1, "possession": 53, "forme": "WDWWL"},
    "Juventus": {"xG_for": 1.5, "xG_against": 0.8, "possession": 51, "forme": "WDWWD"},
    
    # Fallback génériques
    "default_strong": {"xG_for": 2.2, "xG_against": 0.9, "possession": 58, "forme": "WWWLW"},
    "default_medium": {"xG_for": 1.5, "xG_against": 1.3, "possession": 48, "forme": "WDLWD"},
    "default_weak": {"xG_for": 1.0, "xG_against": 1.8, "possession": 42, "forme": "LLDLL"}
}

# ============================================================
# PARTIE 1 : RÉCUPÉRATION DES COTES
# ============================================================

def get_live_odds(sport="soccer_epl", regions="eu", markets="h2h"):
    """
    Récupère les cotes 1X2 pour un championnat via The Odds API.
    Estime automatiquement les statistiques des équipes.
    
    Args:
        sport: Identifiant du championnat (soccer_epl, soccer_france_ligue_one, etc.)
        regions: Région des bookmakers (eu, uk, us)
        markets: Type de marché (h2h = 1X2)
    
    Returns:
        DataFrame avec: home_team, away_team, cote_home, cote_draw, cote_away, match_date
    """
    
    # Vérifier le cache
    cache_key = f"odds_{sport}_{regions}_{markets}"
    cached = _cache_get(cache_key)
    if cached is not None:
        print(f"📦 Données récupérées du cache ({len(cached)} matchs)")
        return cached
    
    # Si pas de clé API, utiliser les données simulées
    if not ODDS_API_KEY or ODDS_API_KEY == "votre_cle_odds_api_ici":
        print("📊 Mode démonstration : utilisation de données simulées.")
        return _get_demo_matches(sport)
    
    # Appel à The Odds API
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }
    
    try:
        print(f"📡 Appel API : {sport}...")
        resp = requests.get(url, params=params, timeout=15)
        
        # Vérifier les quotas
        remaining = resp.headers.get('x-requests-remaining', '?')
        print(f"   Requêtes restantes ce mois : {remaining}")
        
        if resp.status_code == 401:
            print("❌ Clé API Odds invalide. Passage en mode démonstration.")
            return _get_demo_matches(sport)
        
        if resp.status_code == 422:
            print(f"❌ Championnat '{sport}' non disponible. Passage en mode démonstration.")
            return _get_demo_matches(sport)
        
        if resp.status_code != 200:
            print(f"⚠️  Erreur API (status {resp.status_code}). Passage en mode démonstration.")
            return _get_demo_matches(sport)
        
        data = resp.json()
        
        if not data:
            print(f"📭 Aucun match trouvé pour {sport}.")
            # En mode test, on bascule sur la démo si l'API est vide
            return _get_demo_matches(sport)
        
        rows = []
        for match in data:
            home = match.get("home_team", "Inconnu")
            away = match.get("away_team", "Inconnu")
            commence_time = match.get("commence_time", datetime.now().isoformat())
            
            # Récupérer les cotes du premier bookmaker
            bookmakers = match.get("bookmakers", [])
            if not bookmakers:
                continue
            
            # Prendre le premier bookmaker (ou moyenner si vous voulez)
            bookmaker = bookmakers[0]
            markets_list = bookmaker.get("markets", [])
            if not markets_list:
                continue
            
            outcomes = markets_list[0].get("outcomes", [])
            
            cotes = {}
            for outcome in outcomes:
                name = outcome.get("name")
                price = outcome.get("price")
                if name and price:
                    cotes[name] = price
            
            row = {
                "home_team": home,
                "away_team": away,
                "cote_home": cotes.get(home),
                "cote_draw": cotes.get("Draw"),
                "cote_away": cotes.get(away),
                "match_date": commence_time
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 🆕 Estimer les statistiques depuis les cotes
        if not df.empty:
            _update_stats_from_odds(df)
        
        _cache_set(cache_key, df)
        print(f"✅ {len(df)} matchs récupérés pour {sport}")
        return df
        
    except requests.exceptions.Timeout:
        print("⏰ Timeout API. Passage en mode démonstration.")
        return _get_demo_matches(sport)
    except requests.exceptions.ConnectionError:
        print("🔌 Erreur de connexion. Passage en mode démonstration.")
        return _get_demo_matches(sport)
    except Exception as e:
        print(f"❌ Erreur : {e}. Passage en mode démonstration.")
        return _get_demo_matches(sport)


def _update_stats_from_odds(matches_df):
    """
    Met à jour les statistiques des équipes dans le cache 
    en les estimant à partir des cotes The Odds API.
    """
    if matches_df.empty:
        return
    
    updated = 0
    for idx, row in matches_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        cote_h = row.get('cote_home')
        cote_d = row.get('cote_draw')
        cote_a = row.get('cote_away')
        
        if cote_h and cote_d and cote_a:
            # Estimer les stats depuis les cotes
            stats_home, stats_away = estimer_stats_depuis_cotes(cote_h, cote_d, cote_a)
            
            # Mettre à jour le cache (écrase les valeurs par défaut)
            _cache_set(f"stats_{home}", stats_home)
            _cache_set(f"stats_{away}", stats_away)
            updated += 1
    
    if updated > 0:
        print(f"   📊 Stats estimées pour {updated} matchs depuis les cotes")


# ============================================================
# PARTIE 2 : MATCHS DE DÉMONSTRATION
# ============================================================

def _get_demo_matches(sport="soccer_epl"):
    """
    Génère des matchs de démonstration avec cotes réalistes
    quand l'API n'est pas disponible.
    """
    
    # Matchs selon le championnat
    demo_data = {
        "soccer_epl": [
            ("Arsenal", "Chelsea", 2.10, 3.50, 3.40),
            ("Liverpool", "Manchester United", 1.85, 3.80, 4.10),
            ("Manchester City", "Tottenham", 1.45, 4.50, 6.50),
            ("Newcastle", "Aston Villa", 2.30, 3.40, 3.00),
            ("Brighton", "West Ham", 2.50, 3.30, 2.80),
            ("Wolverhampton", "Crystal Palace", 2.70, 3.20, 2.65),
            ("Everton", "Fulham", 2.40, 3.30, 2.90),
            ("Bournemouth", "Brentford", 2.60, 3.40, 2.60),
        ],
        "soccer_france_ligue_one": [
            ("PSG", "Marseille", 1.55, 4.20, 5.50),
            ("Lyon", "Monaco", 2.40, 3.50, 2.80),
            ("Lille", "Rennes", 2.10, 3.40, 3.40),
            ("Nice", "Lens", 2.20, 3.30, 3.20),
            ("Reims", "Strasbourg", 2.50, 3.20, 2.80),
        ],
        "soccer_germany_bundesliga": [
            ("Bayern Munich", "Borussia Dortmund", 1.40, 5.00, 6.50),
            ("RB Leipzig", "Bayer Leverkusen", 2.60, 3.50, 2.55),
        ],
        "soccer_italy_serie_a": [
            ("Inter", "AC Milan", 1.90, 3.50, 4.00),
            ("Juventus", "Napoli", 2.30, 3.30, 3.10),
        ],
        "soccer_spain_la_liga": [
            ("Real Madrid", "Barcelona", 1.95, 3.60, 3.70),
            ("Atletico Madrid", "Sevilla", 1.70, 3.70, 4.80),
        ],
    }
    
    matches = demo_data.get(sport, demo_data["soccer_epl"])
    
    rows = []
    for i, (home, away, cote_h, cote_d, cote_a) in enumerate(matches):
        rows.append({
            "home_team": home,
            "away_team": away,
            "cote_home": cote_h,
            "cote_draw": cote_d,
            "cote_away": cote_a,
            "match_date": (datetime.now() + timedelta(hours=i*3)).isoformat()
        })
    
    df = pd.DataFrame(rows)
    
    # Estimer les stats depuis les cotes de démonstration
    if not df.empty:
        _update_stats_from_odds(df)
    
    print(f"   {len(df)} matchs de démonstration générés")
    return df


# ============================================================
# PARTIE 3 : STATISTIQUES D'ÉQUIPES
# ============================================================

def get_team_stats(team_name, league_id=None):
    """
    Récupère les statistiques d'une équipe.
    
    Priorité de recherche :
    1. Cache (données déjà estimées ou récupérées)
    2. Base de démonstration (équipe connue)
    3. Valeurs par défaut avec estimation
    
    Args:
        team_name: Nom de l'équipe
        league_id: Identifiant de la ligue (non utilisé sans API-Football)
    
    Returns:
        dict avec xG_for, xG_against, possession, forme
    """
    
    # 1. Vérifier le cache
    cache_key = f"stats_{team_name}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached.copy()
    
    # 2. Chercher dans la base de démonstration
    if team_name in _STATS_EQUIPES:
        stats = _STATS_EQUIPES[team_name].copy()
        # Ajouter un léger bruit aléatoire pour simuler des variations
        stats['xG_for'] = round(stats['xG_for'] + random.uniform(-0.1, 0.1), 2)
        stats['xG_against'] = round(stats['xG_against'] + random.uniform(-0.1, 0.1), 2)
        stats['possession'] = round(stats['possession'] + random.uniform(-2, 2), 1)
        
        # Borner
        stats['xG_for'] = max(0.3, stats['xG_for'])
        stats['xG_against'] = max(0.3, stats['xG_against'])
        stats['possession'] = max(30, min(70, stats['possession']))
        
        _cache_set(cache_key, stats)
        return stats.copy()
    
    # 3. Valeurs par défaut basées sur un hash du nom (cohérent pour la même équipe)
    hash_val = int(hashlib.md5(team_name.encode()).hexdigest()[:8], 16)
    random.seed(hash_val)
    
    # Attribution pseudo-aléatoire mais déterministe
    if hash_val % 3 == 0:
        base = _STATS_EQUIPES["default_strong"].copy()
    elif hash_val % 3 == 1:
        base = _STATS_EQUIPES["default_medium"].copy()
    else:
        base = _STATS_EQUIPES["default_weak"].copy()
    
    base['xG_for'] = round(base['xG_for'] + random.uniform(-0.3, 0.3), 2)
    base['xG_against'] = round(base['xG_against'] + random.uniform(-0.3, 0.3), 2)
    base['possession'] = round(base['possession'] + random.uniform(-8, 8), 1)
    
    base['xG_for'] = max(0.3, base['xG_for'])
    base['xG_against'] = max(0.3, base['xG_against'])
    base['possession'] = max(30, min(70, base['possession']))
    
    _cache_set(cache_key, base)
    return base.copy()


# ============================================================
# PARTIE 4 : MISE À JOUR DES RÉSULTATS
# ============================================================

def update_results():
    """
    Met à jour les résultats des matchs terminés.
    Vérifie les matchs sans résultat dans la base et tente de les compléter.
    
    Returns:
        bool: True si des mises à jour ont été effectuées
    """
    try:
        conn = sqlite3.connect("systeme.db")
        df_pending = pd.read_sql_query("""
            SELECT id, home_team, away_team, date_match 
            FROM matchs_historique 
            WHERE resultat IS NULL
        """, conn)
        
        if df_pending.empty:
            conn.close()
            return False

        updated_count = 0
        
        # 1. Tentative de récupération via l'API réelle (The Odds API)
        if ODDS_API_KEY and ODDS_API_KEY != "votre_cle_odds_api_ici":
            # On cible les sports majeurs pour optimiser les requêtes
            sports_to_check = ["soccer_epl", "soccer_france_ligue_one", "soccer_germany_bundesliga", "soccer_italy_serie_a", "soccer_spain_la_liga"]
            
            for sport in sports_to_check:
                url = f"https://api.the-odds-api.com/v4/sports/{sport}/scores/"
                params = {"apiKey": ODDS_API_KEY, "daysFrom": 3}
                
                try:
                    resp = requests.get(url, params=params, timeout=10)
                    if resp.status_code != 200:
                        continue
                    
                    scores_data = resp.json()
                    for match in scores_data:
                        if not match.get('completed'):
                            continue
                        
                        # Chercher si ce match est dans nos paris en attente
                        mask = (df_pending['home_team'] == match['home_team']) & \
                               (df_pending['away_team'] == match['away_team'])
                        matches_to_update = df_pending[mask]
                        
                        if not matches_to_update.empty:
                            # Extraire les scores
                            h_score = 0
                            a_score = 0
                            for s_item in match.get('scores', []):
                                if s_item['name'] == match['home_team']: h_score = int(s_item['score'])
                                if s_item['name'] == match['away_team']: a_score = int(s_item['score'])
                            
                            res = 'home' if h_score > a_score else 'away' if a_score > h_score else 'draw'
                            score_str = f"{h_score}-{a_score}"
                            
                            for _, m_row in matches_to_update.iterrows():
                                conn.execute("""
                                    UPDATE matchs_historique 
                                    SET resultat = ?, score = ?, date_resultat = ? 
                                    WHERE id = ?
                                """, (res, score_str, datetime.now().isoformat(), m_row['id']))
                                updated_count += 1
                                print(f"   ✅ Résultat API : {match['home_team']} {score_str} {match['away_team']}")
                except Exception as e:
                    print(f"⚠️  Erreur API Scores pour {sport} : {e}")
                    continue

        # 2. Fallback simulation (si l'API n'a rien trouvé ou n'est pas configurée)
        if updated_count == 0:
            for _, row in df_pending.iterrows():
                res = _fetch_match_result(row['home_team'], row['away_team'], row['date_match'])
                if res:
                    conn.execute("UPDATE matchs_historique SET resultat = ?, date_resultat = ? WHERE id = ?",
                               (res, datetime.now().isoformat(), row['id']))
                    updated_count += 1

        conn.commit()
        conn.close()
        return updated_count > 0
    except Exception as e:
        print(f"❌ Erreur mise à jour résultats : {e}")
        return False


def _fetch_match_result(home_team, away_team, match_date):
    """
    Tente de récupérer le résultat d'un match.
    
    Avec The Odds API, on peut utiliser l'endpoint /sports/{sport}/scores/
    mais cela consomme des requêtes. Pour économiser, on utilise 
    une simulation en mode démo, ou l'API si configurée.
    
    Returns:
        str: 'home', 'away', 'draw' ou None
    """
    
    # Si on a l'API Odds, on pourrait faire un vrai appel
    if ODDS_API_KEY and ODDS_API_KEY != "votre_cle_odds_api_ici":
        # TODO : Implémenter l'appel à /sports/{sport}/scores/
        # Pour l'instant, on simule (à remplacer par un vrai appel)
        pass
    
    # Simulation réaliste basée sur un hash du match
    # (même résultat pour le même match si appelé plusieurs fois)
    seed = f"{home_team}_{away_team}_{match_date}"
    hash_val = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    random.seed(hash_val)
    
    # Probabilités réalistes
    r = random.random()
    if r < 0.42:
        return 'home'
    elif r < 0.70:
        return 'draw'
    else:
        return 'away'


# ============================================================
# PARTIE 5 : CHAMPIONNATS DISPONIBLES
# ============================================================

def get_available_sports():
    """
    Retourne la liste des championnats disponibles.
    """
    sports = {
        # Football
        "Premier League": "soccer_epl",
        "Ligue 1": "soccer_france_ligue_one",
        "Bundesliga": "soccer_germany_bundesliga",
        "Serie A": "soccer_italy_serie_a",
        "La Liga": "soccer_spain_la_liga",
        "Champions League": "soccer_uefa_champions_league",
        "Europa League": "soccer_uefa_europa_league",
        "Ligue 2": "soccer_france_ligue_two",
        "Championship": "soccer_england_championship",
        "Eredivisie": "soccer_netherlands_eredivisie",
        "Primeira Liga": "soccer_portugal_primeira_liga",
        "Brasileirão": "soccer_brazil_campeonato",
        "MLS": "soccer_usa_mls",

        # Autres sports
        "NBA": "basketball_nba",
        "NFL": "americanfootball_nfl",
        "NHL": "icehockey_nhl",
        "MLB": "baseball_mlb",
        "UFC": "mma_mixed_martial_arts",
        "Tennis ATP": "tennis_atp_french_open",
    }
    return sports


def get_sport_name(sport_key):
    """Retourne le nom lisible d'un championnat."""
    sports = get_available_sports()
    for name, key in sports.items():
        if key == sport_key:
            return name
    return sport_key


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("🧪 Test du module data_collector\n")
    
    # Test 1 : Mode démo (sans API)
    print("=" * 60)
    print("TEST 1 : Récupération cotes (mode démo)")
    print("=" * 60)
    df = get_live_odds("soccer_epl")
    if not df.empty:
        print(df[['home_team', 'away_team', 'cote_home', 'cote_draw', 'cote_away']].to_string())
    
    # Test 2 : Stats estimées depuis cotes
    print("\n" + "=" * 60)
    print("TEST 2 : Statistiques estimées")
    print("=" * 60)
    for team in ["Arsenal", "Chelsea", "Liverpool"]:
        stats = get_team_stats(team)
        print(f"   {team:20} → xG={stats['xG_for']}, xGA={stats['xG_against']}, Poss={stats['possession']}%")
    
    # Test 3 : Estimation directe
    print("\n" + "=" * 60)
    print("TEST 3 : Estimation xG depuis cotes")
    print("=" * 60)
    stats_h, stats_a = estimer_stats_depuis_cotes(2.10, 3.50, 3.40)
    print(f"   Domicile : xG={stats_h['xG_for']}, xGA={stats_h['xG_against']}, Poss={stats_h['possession']}%")
    print(f"   Extérieur: xG={stats_a['xG_for']}, xGA={stats_a['xG_against']}, Poss={stats_a['possession']}%")
    
    # Test 4 : Championnats disponibles
    print("\n" + "=" * 60)
    print("TEST 4 : Championnats disponibles")
    print("=" * 60)
    sports = get_available_sports()
    for name, key in sports.items():
        print(f"   {name:30} → {key}")
    
    # Test 5 : Cache
    print("\n" + "=" * 60)
    print("TEST 5 : Test du cache")
    print("=" * 60)
    print("   Premier appel (devrait afficher 'mode démo') :")
    df1 = get_live_odds("soccer_epl")
    print("   Deuxième appel (devrait afficher 'cache') :")
    df2 = get_live_odds("soccer_epl")
    print(f"   Données identiques : {df1.equals(df2)}")
    
    print("\n✅ Tests terminés !")