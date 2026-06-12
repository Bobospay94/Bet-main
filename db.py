"""
Database Manager - Gestion de la base de données
==================================================

Ce module gère toute la persistance des données :
1. Coefficients tactiques du modèle
2. Historique des matchs analysés
3. Suivi des paris placés
4. Performance quotidienne
5. Configuration du système

Base de données : SQLite (fichier systeme.db)
"""

import sqlite3
import pandas as pd
from datetime import datetime
import json
import os
import numpy as np
from sqlalchemy import create_engine, text
import streamlit as st

DB_NAME = "systeme.db"

def get_engine():
    """
    Crée un moteur SQLAlchemy. 
    Cherche d'abord une DATABASE_URL dans les secrets (PostgreSQL), 
    sinon utilise SQLite local.
    """
    db_url = None

    # 1. Tenter de récupérer depuis Streamlit Secrets (Cloud)
    try:
        db_url = st.secrets.get("DATABASE_URL")
    except:
        pass

    # 2. Tenter de récupérer depuis config.py (Local)
    if not db_url:
        try:
            from config import DATABASE_URL
            db_url = DATABASE_URL
        except ImportError:
            pass
    
    if db_url:
        # Correction pour les URL Heroku/Supabase qui commencent par postgres://
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    
    # Fallback local SQLite
    return create_engine(f"sqlite:///{DB_NAME}")

# ============================================================
# INITIALISATION DE LA BASE DE DONNÉES
# ============================================================

def init_db():
    """
    Crée toutes les tables si elles n'existent pas.
    À appeler au démarrage de l'application.
    """
    engine = get_engine()

    # Utilisation de SQLAlchemy pour la création initiale
    with engine.begin() as conn:
        # --- Table des coefficients tactiques ---
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS coefficients (
                id INTEGER PRIMARY KEY,
                bonus_ailes_pressing REAL DEFAULT 0.20,
                bonus_ailes_bloc REAL DEFAULT 0.40,
                bonus_axe_pressing REAL DEFAULT 0.00,
                bonus_axe_bloc REAL DEFAULT 0.10,
                intercept_logistic REAL DEFAULT 0.00,
                derniere_maj TEXT
            )
        '''))

        # Insertion du record par défaut (syntaxe compatible)
        res = conn.execute(text("SELECT count(*) FROM coefficients")).fetchone()
        if res[0] == 0:
            conn.execute(text("INSERT INTO coefficients (id, derniere_maj) VALUES (1, :maj)"),
                         {"maj": datetime.now().isoformat()})

        # --- Table de l'historique des matchs ---
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS matchs_historique (
                id SERIAL PRIMARY KEY,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                date_match TEXT,
                xg_home REAL,
                xg_away REAL,
                possession_home REAL,
                possession_away REAL,
                strategies TEXT,
                prob_estimee_home REAL,
                cote_home REAL,
                cote_draw REAL,
                cote_away REAL,
                resultat TEXT,
                score TEXT,
                source_stats TEXT DEFAULT 'estimated',
                date_analyse TEXT,
                date_resultat TEXT
            )
        '''))

        # --- Table des paris placés ---
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS paris (
                id SERIAL PRIMARY KEY,
                match_id INTEGER,
                match_nom TEXT NOT NULL,
                date_paris TEXT NOT NULL,
                type_paris TEXT DEFAULT 'home',
                prob_est REAL NOT NULL,
                cote REAL NOT NULL,
                mise REAL NOT NULL,
                resultat TEXT DEFAULT 'Pending',
                gain REAL DEFAULT 0.0,
                date_resultat TEXT,
                notes TEXT
            )
        '''))

        # --- Table de performance quotidienne ---
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS performance_quotidienne (
                date TEXT PRIMARY KEY,
                nb_paris INTEGER DEFAULT 0,
                nb_wins INTEGER DEFAULT 0,
                nb_losses INTEGER DEFAULT 0,
                mise_totale REAL DEFAULT 0.0,
                gain_total REAL DEFAULT 0.0,
                profit REAL DEFAULT 0.0,
                roi REAL DEFAULT 0.0
            )
        '''))

        # --- Table de configuration ---
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS configuration (
                cle TEXT PRIMARY KEY,
                valeur TEXT
            )
        '''))

        # Paramètres par défaut
        defaults = {
            'bankroll': '1000.0',
            'kelly_fraction': '0.25',
            'seuil_value': '0.02',
            'championnat_par_defaut': 'soccer_epl',
            'version_modele': '2.0',
            'max_stake_pct': '0.10'
        }
        for cle, valeur in defaults.items():
            conn.execute(text("INSERT INTO configuration (cle, valeur) VALUES (:cle, :val) ON CONFLICT (cle) DO NOTHING"),
                         {"cle": cle, "val": valeur})

    print("✅ Base de données initialisée via SQLAlchemy.")

# ============================================================
# GESTION DES COEFFICIENTS DU MODÈLE
# ============================================================

def get_coefficients():
    """
    Récupère les coefficients tactiques actuels.
    
    Returns:
        dict avec bonus_ailes_pressing, bonus_ailes_bloc, 
        bonus_axe_pressing, bonus_axe_bloc, intercept
    """
    try:
        with get_engine().connect() as conn:
            row = pd.read_sql_query(text("SELECT * FROM coefficients WHERE id = 1"), conn).iloc[0]
            coeffs = {
                "bonus_ailes_pressing": float(row["bonus_ailes_pressing"]),
                "bonus_ailes_bloc": float(row["bonus_ailes_bloc"]),
                "bonus_axe_pressing": float(row["bonus_axe_pressing"]),
                "bonus_axe_bloc": float(row["bonus_axe_bloc"]),
                "intercept": float(row["intercept_logistic"])
            }
        return coeffs
    except Exception as e:
        print(f"⚠️  Erreur récupération coefficients : {e}")
        # Valeurs par défaut
        return {
            "bonus_ailes_pressing": 0.20,
            "bonus_ailes_bloc": 0.40,
            "bonus_axe_pressing": 0.00,
            "bonus_axe_bloc": 0.10,
            "intercept": 0.00
        }


def update_coefficients(params):
    """
    Met à jour les coefficients tactiques après recalibrage.
    
    Args:
        params: dict avec les nouveaux coefficients
    """
    try:
        with get_engine().begin() as conn:
            conn.execute(text('''
                UPDATE coefficients SET 
                    bonus_ailes_pressing = :bap,
                    bonus_ailes_bloc = :bab,
                    bonus_axe_pressing = :bxp,
                    bonus_axe_bloc = :bxb,
                    intercept_logistic = :int,
                    derniere_maj = :maj
                WHERE id = 1
            '''), {
                "bap": params.get("bonus_ailes_pressing", 0.20),
                "bab": params.get("bonus_ailes_bloc", 0.40),
                "bxp": params.get("bonus_axe_pressing", 0.00),
                "bxb": params.get("bonus_axe_bloc", 0.10),
                "int": params.get("intercept", 0.00),
                "maj": datetime.now().isoformat()
            })
        print("✅ Coefficients mis à jour.")
    except Exception as e:
        print(f"❌ Erreur mise à jour coefficients : {e}")


def get_coefficients_history():
    """Retourne l'historique de la dernière mise à jour des coefficients."""
    with get_engine().connect() as conn:
        row = pd.read_sql_query(text("SELECT *, derniere_maj FROM coefficients WHERE id = 1"), conn).iloc[0]
        return {
            "coefficients": {
                "bonus_ailes_pressing": row["bonus_ailes_pressing"],
                "bonus_ailes_bloc": row["bonus_ailes_bloc"],
                "bonus_axe_pressing": row["bonus_axe_pressing"],
                "bonus_axe_bloc": row["bonus_axe_bloc"],
                "intercept": row["intercept_logistic"]
            },
            "derniere_maj": row["derniere_maj"]
        }


# ============================================================
# GESTION DE L'HISTORIQUE DES MATCHS
# ============================================================

def ajouter_match_historique(home_team, away_team, date_match, xg_home, xg_away,
                              possession_home, possession_away, strategies,
                              prob_estimee_home, cote_home, cote_draw, cote_away,
                              resultat=None, score=None, source_stats='estimated'):
    """
    Ajoute un match à l'historique pour le recalibrage futur.
    
    Args:
        home_team: Équipe à domicile
        away_team: Équipe à l'extérieur
        date_match: Date/heure du match
        xg_home: xG estimé pour l'équipe à domicile
        xg_away: xG estimé pour l'équipe à l'extérieur
        possession_home: Possession estimée domicile
        possession_away: Possession estimée extérieur
        strategies: dict {'A': 'ailes', 'B': 'pressing_haut'}
        prob_estimee_home: Notre probabilité estimée
        cote_home, cote_draw, cote_away: Cotes bookmaker
        resultat: 'home', 'away', 'draw' ou None
        score: '2-1' par exemple
        source_stats: 'estimated' (depuis cotes) ou 'api' (API-Football)
    
    Returns:
        int: l'ID du match inséré
    """
    try:
        with get_engine().begin() as conn:
            res = conn.execute(text('''
            INSERT INTO matchs_historique 
            (home_team, away_team, date_match, xg_home, xg_away, 
             possession_home, possession_away, strategies,
             prob_estimee_home, cote_home, cote_draw, cote_away,
             resultat, score, source_stats, date_analyse)
            VALUES (:ht, :at, :dm, :xh, :xa, :ph, :pa, :st, :peh, :ch, :cd, :ca, :res, :sc, :src, :da) 
            RETURNING id
            '''), {
                "ht": home_team, "at": away_team, "dm": date_match,
                "xh": xg_home, "xa": xg_away,
                "ph": possession_home, "pa": possession_away,
                "st": json.dumps(strategies),
                "peh": prob_estimee_home, "ch": cote_home, "cd": cote_draw, "ca": cote_away,
                "res": resultat, "sc": score, "src": source_stats,
                "da": datetime.now().isoformat()
            })
        return res.fetchone()[0]
    except Exception as e:
        print(f"❌ Erreur ajout match historique : {e}")
        return None

def mettre_a_jour_resultat(match_id, resultat, score=None):
    """
    Met à jour le résultat d'un match déjà enregistré.
    
    Args:
        match_id: ID du match
        resultat: 'home', 'away' ou 'draw'
        score: '2-1' par exemple
    """
    try:
        with get_engine().begin() as conn:
            conn.execute(text('''
            UPDATE matchs_historique 
            SET resultat = :res, score = :sc, date_resultat = :dr
            WHERE id = :id
            '''), {
                "res": resultat, "sc": score, 
                "dr": datetime.now().isoformat(), "id": match_id
            })
        print(f"✅ Résultat mis à jour pour le match {match_id} : {resultat}")
    except Exception as e:
        print(f"❌ Erreur mise à jour résultat : {e}")


def get_matchs_sans_resultat():
    """Récupère les matchs en attente de résultat."""
    with get_engine().connect() as conn:
        return pd.read_sql_query(text("SELECT * FROM matchs_historique WHERE resultat IS NULL"), conn)


def get_matchs_avec_resultat():
    """Récupère tous les matchs qui ont un résultat (pour le recalibrage)."""
    with get_engine().connect() as conn:
        return pd.read_sql_query(
            text("SELECT * FROM matchs_historique WHERE resultat IS NOT NULL ORDER BY date_match DESC"),
            conn
        )


def get_matchs_recents(n=20):
    """Récupère les n matchs les plus récents."""
    with get_engine().connect() as conn:
        return pd.read_sql_query(
            text(f"SELECT * FROM matchs_historique ORDER BY date_analyse DESC LIMIT {n}"),
            conn
        )


def get_nb_matchs_historique():
    """Retourne le nombre total de matchs dans l'historique."""
    with get_engine().connect() as conn:
        res = conn.execute(text("SELECT COUNT(*) FROM matchs_historique")).fetchone()
        return res[0]


def get_stats_source_distribution():
    """Retourne la répartition des sources de statistiques."""
    with get_engine().connect() as conn:
        return pd.read_sql_query(text("""
            SELECT source_stats, COUNT(*) as nb 
            FROM matchs_historique 
            GROUP BY source_stats
        """), conn)


def get_match_details(match_id):
    """
    Récupère le score et le résultat d'un match par son ID.
    """
    with get_engine().connect() as conn:
        row = conn.execute(text("SELECT score, resultat FROM matchs_historique WHERE id = :id"), {"id": match_id}).fetchone()
        return {"score": row[0], "resultat": row[1]} if row else {"score": None, "resultat": None}

# ============================================================
# GESTION DES PARIS
# ============================================================

def enregistrer_paris(match, prob_est, cote, mise, type_paris="home", 
                      match_id=None, resultat="Pending", gain=0.0, notes=""):
    """
    Enregistre un nouveau pari dans le suivi.

    Args:
        match: Nom du match (ex: "Arsenal vs Chelsea")
        prob_est: Notre probabilité estimée
        cote: Cote prise
        mise: Montant misé
        type_paris: 'home', 'draw' ou 'away'
        match_id: ID du match dans matchs_historique (optionnel)
        resultat: 'Pending', 'Win', 'Loss', 'Void'
        gain: Gain réalisé (0 tant que pending)
        notes: Notes éventuelles
    
    Returns:
        int: l'ID du pari inséré
    """
    try:
        with get_engine().begin() as conn:
            res = conn.execute(text('''
            INSERT INTO paris 
            (match_id, match_nom, date_paris, type_paris, prob_est, cote, mise, 
             resultat, gain, notes)
            VALUES (:mid, :mn, :dp, :tp, :pe, :ct, :ms, :res, :gn, :nt) 
            RETURNING id
            '''), {
                "mid": match_id, "mn": match, "dp": datetime.now().isoformat(),
                "tp": type_paris, "pe": prob_est, "ct": cote, "ms": mise,
                "res": resultat, "gn": gain, "nt": notes
            })
        return res.fetchone()[0]
    except Exception as e:
        print(f"❌ Erreur enregistrement pari : {e}")
        return None


def mettre_a_jour_paris(pari_id, resultat, gain):
    """
    Met à jour le résultat d'un pari.
    
    Args:
        pari_id: ID du pari
        resultat: 'Win', 'Loss', 'Void'
        gain: Montant gagné (0 si perdu)
    """
    try:
        with get_engine().begin() as conn:
            conn.execute(text('''
            UPDATE paris 
            SET resultat = :res, gain = :gn, date_resultat = :dr
            WHERE id = :id
            '''), {
                "res": resultat, "gn": gain, 
                "dr": datetime.now().isoformat(), "id": pari_id
            })
            # Mettre à jour la performance quotidienne
            _update_performance_quotidienne(conn)
        print(f"✅ Pari {pari_id} mis à jour : {resultat} - Gain: {gain:.2f}€")
    except Exception as e:
        print(f"❌ Erreur mise à jour pari : {e}")


def get_paris_en_attente():
    """Récupère les paris dont le résultat n'est pas encore connu."""
    with get_engine().connect() as conn:
        return pd.read_sql_query(text("SELECT * FROM paris WHERE resultat = 'Pending' ORDER BY date_paris DESC"), conn)


def get_tous_les_paris(limite=100):
    """Récupère les derniers paris."""
    with get_engine().connect() as conn:
        return pd.read_sql_query(text(f"SELECT * FROM paris ORDER BY date_paris DESC LIMIT {limite}"), conn)


def get_paris_termines():
    """Récupère les paris qui ont un résultat définitif."""
    with get_engine().connect() as conn:
        return pd.read_sql_query(text("SELECT * FROM paris WHERE resultat != 'Pending' ORDER BY date_paris DESC"), conn)


def get_paris_du_jour():
    """Récupère les paris placés aujourd'hui."""
    aujourdhui = datetime.now().strftime('%Y-%m-%d')
    with get_engine().connect() as conn:
        return pd.read_sql_query(text(f"SELECT * FROM paris WHERE date(date_paris) = '{aujourdhui}' ORDER BY date_paris DESC"), conn)


def get_statistiques_paris():
    """
    Calcule les statistiques globales des paris.

    Returns:
        dict avec nb_paris, nb_wins, win_rate, profit_total, roi, etc.
    """
    with get_engine().connect() as conn:
        df = pd.read_sql_query(
            text("SELECT * FROM paris WHERE resultat != 'Pending'"), conn
        )
        
        if df.empty:
            return {"nb_paris": 0, "nb_wins": 0, "nb_losses": 0, "win_rate": 0, "mise_totale": 0, "gain_total": 0, "profit_total": 0, "roi": 0, "cote_moyenne": 0, "cote_moyenne_gagnante": 0, "meilleure_cote": 0}
        
        nb_paris = len(df)
        nb_wins = len(df[df['resultat'] == 'Win'])
        nb_losses = len(df[df['resultat'] == 'Loss'])
        win_rate = (nb_wins / nb_paris * 100) if nb_paris > 0 else 0

        mise_totale = df['mise'].sum()
        gain_total = df['gain'].sum()
        profit_total = gain_total - mise_totale
        roi = (profit_total / mise_totale * 100) if mise_totale > 0 else 0
        
        cote_moyenne = df['cote'].mean()
        cote_moyenne_gagnante = df[df['resultat'] == 'Win']['cote'].mean() if nb_wins > 0 else 0
        meilleure_cote = df[df['resultat'] == 'Win']['cote'].max() if nb_wins > 0 else 0

        return {
            "nb_paris": nb_paris,
            "nb_wins": nb_wins,
            "nb_losses": nb_losses,
            "win_rate": round(win_rate, 1),
            "mise_totale": round(mise_totale, 2),
            "gain_total": round(gain_total, 2),
            "profit_total": round(profit_total, 2),
            "roi": round(roi, 1),
            "cote_moyenne": round(cote_moyenne, 2),
            "cote_moyenne_gagnante": round(cote_moyenne_gagnante, 2),
            "meilleure_cote": round(meilleure_cote, 2)
        }


def _update_performance_quotidienne(conn):
    """Recalcule les performances quotidiennes (appelé en interne)."""
    try:
        aujourdhui = datetime.now().strftime('%Y-%m-%d')

        df = pd.read_sql_query(f"""
            SELECT * FROM paris 
            WHERE date(date_paris) = '{aujourdhui}' 
            AND resultat != 'Pending'
        """, conn) # SQLAlchemy connection passed from caller

        if df.empty:
            return

        nb_paris = len(df)
        nb_wins = len(df[df['resultat'] == 'Win'])
        nb_losses = len(df[df['resultat'] == 'Loss'])
        mise_totale = df['mise'].sum()
        gain_total = df['gain'].sum()
        profit = gain_total - mise_totale
        roi = (profit / mise_totale * 100) if mise_totale > 0 else 0

        conn.execute(text('''
            INSERT INTO performance_quotidienne 
            (date, nb_paris, nb_wins, nb_losses, mise_totale, gain_total, profit, roi)
            VALUES (:date, :nbp, :nbw, :nbl, :mt, :gt, :p, :roi)
            ON CONFLICT (date) DO UPDATE SET nb_paris=EXCLUDED.nb_paris, nb_wins=EXCLUDED.nb_wins, nb_losses=EXCLUDED.nb_losses, mise_totale=EXCLUDED.mise_totale, gain_total=EXCLUDED.gain_total, profit=EXCLUDED.profit, roi=EXCLUDED.roi
        '''), {"date": aujourdhui, "nbp": nb_paris, "nbw": nb_wins, "nbl": nb_losses, "mt": mise_totale, "gt": gain_total, "p": profit, "roi": roi})
    except Exception as e:
        print(f"⚠️  Erreur mise à jour performance quotidienne : {e}")


def get_performance_quotidienne(jours=30):
    """Récupère les performances des n derniers jours."""
    with get_engine().connect() as conn:
        return pd.read_sql_query(text(f"SELECT * FROM performance_quotidienne ORDER BY date DESC LIMIT {jours}"), conn)


# ============================================================
# GESTION DE LA CONFIGURATION
# ============================================================

def get_config(cle):
    """Récupère une valeur de configuration."""
    with get_engine().connect() as conn:
        row = conn.execute(text("SELECT valeur FROM configuration WHERE cle = :cle"), {"cle": cle}).fetchone()
        return row[0] if row else None


def set_config(cle, valeur):
    """Définit une valeur de configuration."""
    with get_engine().begin() as conn:
        conn.execute(text("INSERT INTO configuration (cle, valeur) VALUES (:cle, :val) ON CONFLICT (cle) DO UPDATE SET valeur = EXCLUDED.valeur"),
                     {"cle": cle, "val": str(valeur)})


def get_all_config():
    """Récupère toute la configuration sous forme de dictionnaire."""
    with get_engine().connect() as conn:
        df = pd.read_sql_query(text("SELECT * FROM configuration"), conn)
        return dict(zip(df['cle'], df['valeur']))


# ============================================================
# RÉINITIALISATION
# ============================================================

def reset_database():
    """
    Supprime et recrée toute la base de données.
    ⚠️  ATTENTION : toutes les données seront perdues !
    """
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"🗑️  Base de données '{DB_NAME}' supprimée.")
    init_db()
    print("✅ Base de données réinitialisée.")


# ============================================================
# EXPORT / IMPORT
# ============================================================

def exporter_paris_csv(fichier="export_paris.csv"):
    """Exporte tous les paris en CSV."""
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM paris", conn)
        df.to_csv(fichier, index=False, encoding='utf-8')
        print(f"✅ {len(df)} paris exportés vers {fichier}")
        return fichier
    finally:
        conn.close()


def exporter_matchs_csv(fichier="export_matchs.csv"):
    """Exporte l'historique des matchs en CSV."""
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM matchs_historique", conn)
        df.to_csv(fichier, index=False, encoding='utf-8')
        print(f"✅ {len(df)} matchs exportés vers {fichier}")
        return fichier
    finally:
        conn.close()


def importer_matchs_csv(fichier):
    """Importe des matchs depuis un fichier CSV."""
    try:
        df = pd.read_csv(fichier)
        conn = sqlite3.connect(DB_NAME)
        df.to_sql('matchs_historique', conn, if_exists='append', index=False)
        conn.close()
        print(f"✅ {len(df)} matchs importés depuis {fichier}")
        return True
    except Exception as e:
        print(f"❌ Erreur import : {e}")
        return False


# ============================================================
# STATISTIQUES DE LA BASE
# ============================================================

def get_db_stats():
    """Retourne des statistiques générales sur la base de données."""
    conn = sqlite3.connect(DB_NAME)
    try:
        stats = {}
        
        # Nombre de matchs
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matchs_historique")
        stats['nb_matchs'] = cursor.fetchone()[0]
        
        # Nombre de matchs avec résultat
        cursor.execute("SELECT COUNT(*) FROM matchs_historique WHERE resultat IS NOT NULL")
        stats['nb_matchs_resultats'] = cursor.fetchone()[0]
        
        # Nombre de paris
        cursor.execute("SELECT COUNT(*) FROM paris")
        stats['nb_paris'] = cursor.fetchone()[0]
        
        # Nombre de paris terminés
        cursor.execute("SELECT COUNT(*) FROM paris WHERE resultat != 'Pending'")
        stats['nb_paris_termines'] = cursor.fetchone()[0]
        
        # Taille de la base
        if os.path.exists(DB_NAME):
            stats['taille_ko'] = round(os.path.getsize(DB_NAME) / 1024, 1)
        
        return stats
    finally:
        conn.close()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("🧪 Test du module db\n")
    
    # Initialisation
    init_db()
    
    # Test coefficients
    print("=" * 60)
    print("TEST 1 : Coefficients actuels")
    print("=" * 60)
    coeffs = get_coefficients()
    for k, v in coeffs.items():
        print(f"   {k}: {v:.4f}")
    
    # Test ajout match
    print("\n" + "=" * 60)
    print("TEST 2 : Ajout d'un match test")
    print("=" * 60)
    match_id = ajouter_match_historique(
        home_team="Arsenal",
        away_team="Chelsea",
        date_match=datetime.now().isoformat(),
        xg_home=2.1,
        xg_away=1.3,
        possession_home=58,
        possession_away=42,
        strategies={"A": "ailes", "B": "pressing_haut"},
        prob_estimee_home=0.62,
        cote_home=2.10,
        cote_draw=3.50,
        cote_away=3.40,
        source_stats='estimated'
    )
    print(f"   Match ajouté avec ID = {match_id}")
    
    # Test enregistrement pari
    print("\n" + "=" * 60)
    print("TEST 3 : Enregistrement d'un pari")
    print("=" * 60)
    pari_id = enregistrer_paris(
        match="Arsenal vs Chelsea",
        prob_est=0.62,
        cote=2.10,
        mise=25.0,
        match_id=match_id
    )
    print(f"   Pari ajouté avec ID = {pari_id}")
    
    # Test mise à jour résultat
    print("\n" + "=" * 60)
    print("TEST 4 : Mise à jour du résultat")
    print("=" * 60)
    mettre_a_jour_resultat(match_id, "home", "2-1")
    mettre_a_jour_paris(pari_id, "Win", 52.50)
    
    # Statistiques
    print("\n" + "=" * 60)
    print("TEST 5 : Statistiques des paris")
    print("=" * 60)
    stats = get_statistiques_paris()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    # Configuration
    print("\n" + "=" * 60)
    print("TEST 6 : Configuration")
    print("=" * 60)
    config = get_all_config()
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    # Stats base
    print("\n" + "=" * 60)
    print("TEST 7 : Statistiques de la base")
    print("=" * 60)
    db_stats = get_db_stats()
    for k, v in db_stats.items():
        print(f"   {k}: {v}")
    
    # Export
    print("\n" + "=" * 60)
    print("TEST 8 : Export CSV")
    print("=" * 60)
    exporter_paris_csv("test_export_paris.csv")
    exporter_matchs_csv("test_export_matchs.csv")
    
    # Nettoyage des fichiers de test
    for f in ["test_export_paris.csv", "test_export_matchs.csv"]:
        if os.path.exists(f):
            os.remove(f)
    
    print("\n✅ Tests terminés !")