import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import sys
import os

# Ajouter le répertoire courant au path pour les imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import init_db, get_coefficients, DB_NAME
from data_collector import get_live_odds, get_team_stats, get_available_sports
from matrix_builder import build_payoff_matrix
from nash_equilibrium import solve_nash_2x2
from value_detector import compute_kelly_fraction, find_value_bets, combine_bets
from db import get_match_details, ajouter_match_historique, enregistrer_paris, get_statistiques_paris, get_paris_termines, get_performance_quotidienne, get_config # Import de la nouvelle fonction
from recalibrator import recalibrate

try:
    from config import MODE
except ImportError:
    MODE = st.secrets.get("MODE", "production")

# --- Custom CSS for a cleaner look (minimalist) ---
st.markdown("""
<style>
    /* General container padding */
    .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    /* Custom button style */
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        border: 1px solid #4CAF50; /* Green border */
        color: #4CAF50; /* Green text */
        background-color: white;
        padding: 0.6rem 1rem;
        font-size: 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #4CAF50; /* Green background on hover */
        color: white;
    }
    /* Expander styling */
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration de la page ---
st.set_page_config(
    page_title="Pronostics Value Bets - Théorie des Jeux",
    page_icon="⚽",
    layout="wide"
)
# --- Header ---
st.title("⚽ Bet-main : Système de Pronostics Sportifs")
st.markdown("---")
st.markdown("""
    Bienvenue sur votre tableau de bord de pronostics sportifs.
    Ce système utilise la **Théorie des Jeux** et l'**Équilibre de Nash** pour estimer les probabilités de victoire,
    détecter les **Value Bets** et optimiser les mises via le **Critère de Kelly**.
    Le modèle se **recalibre automatiquement** pour une amélioration continue.
""")

def format_match_datetime(value):
    """Formate une date ISO de match pour l'affichage dans l'interface."""
    if not value:
        return "-"

    try:
        match_dt = pd.to_datetime(value, utc=True)
        return match_dt.tz_convert("Europe/Paris").strftime("%d/%m/%Y %H:%M")
    except Exception:
        try:
            return pd.to_datetime(value).strftime("%d/%m/%Y %H:%M")
        except Exception:
            return str(value)


def extract_match_day(value):
    """Extrait le jour (timezone Europe/Paris) depuis une date ISO de match."""
    if not value:
        return None

    try:
        match_dt = pd.to_datetime(value, utc=True)
        return match_dt.tz_convert("Europe/Paris").date()
    except Exception:
        try:
            return pd.to_datetime(value).date()
        except Exception:
            return None


def format_fcfa(value):
    """Formate un montant en FCFA (sans décimales)."""
    if value is None or pd.isna(value):
        return "-"
    try:
        amount = int(round(float(value)))
        return f"{amount:,}".replace(",", " ") + " FCFA"
    except Exception:
        return "-"

# Initialiser la base de données au démarrage
init_db()

# --- Barre latérale ---
st.sidebar.title("⚙️ Configuration & Actions")

# Liste des sports disponible depuis la source centralisée
sport_map = get_available_sports() # Assuming this function exists and works
sport_names = list(sport_map.keys())

analysis_scope = st.sidebar.radio(
    "Mode d'analyse",
    ["Sport selectionne", "Tous les sports"],
    index=0,
)

# Sélection du sport/compétition (uniquement en mode sport unique)
if analysis_scope == "Sport selectionne":
    championnat = st.sidebar.selectbox(
        "Sport / compétition",
        sport_names,
        index=0,
    )
else:
    championnat = "Tous les sports"

# Paramètres bankroll / risque
st.sidebar.header("💰 Profil de Risque")

bankroll = st.sidebar.number_input(
    "Capital total (FCFA) :",
    min_value=10.0,
    value=1000.0, # Changed default to 1000 for more realistic Kelly calculations
    step=10.0,
)

profil_risque = st.sidebar.selectbox(
    "Choisissez votre profil",
    [
        "🟢 Prudent (recommandé)",
        "🟡 Équilibré",
        "🟠 Offensif",
    ],
    index=0,
)

if profil_risque == "🟢 Prudent (recommandé)":
    KELLY_SIMPLE = 0.20
    KELLY_COMBINE_2 = 0.10
    KELLY_COMBINE_3 = 0.05
    MAX_STAKE_PCT = 0.05
    SEUIL_EV = 0.03
    DESCRIPTION = "🟢 **Prudent** : Peu de paris, risque faible, croissance lente"
elif profil_risque == "🟡 Équilibré":
    KELLY_SIMPLE = 0.25
    KELLY_COMBINE_2 = 0.15
    KELLY_COMBINE_3 = 0.08
    MAX_STAKE_PCT = 0.08
    SEUIL_EV = 0.02
    DESCRIPTION = "🟡 **Équilibré** : Bon compromis risque/rendement"
else:
    KELLY_SIMPLE = 0.35
    KELLY_COMBINE_2 = 0.20
    KELLY_COMBINE_3 = 0.12
    MAX_STAKE_PCT = 0.10
    SEUIL_EV = 0.01
    DESCRIPTION = "🟠 **Offensif** : Plus de paris, gains rapides, risque élevé"

st.sidebar.caption(DESCRIPTION)

with st.sidebar.expander("📊 Résumé des paramètres"):
    st.markdown(
        f"""
        | Paramètre | Valeur |
        |-----------|--------|
        | Kelly paris simples | **{KELLY_SIMPLE:.0%}** |
        | Kelly combiné 2 | **{KELLY_COMBINE_2:.0%}** |
        | Kelly combiné 3 | **{KELLY_COMBINE_3:.0%}** |
        | Mise max par pari | **{MAX_STAKE_PCT:.0%}** du bankroll |
        | Seuil EV minimum | **{SEUIL_EV:.0%}** |
        """
    )

    st.sidebar.markdown(
        f"""
        **Avec {format_fcfa(bankroll)} de bankroll :**
        - Mise max simple : **{format_fcfa(bankroll * MAX_STAKE_PCT)}**
        - Mise max combiné : **{format_fcfa(bankroll * MAX_STAKE_PCT * 0.5)}**
        """
    )

# --- Main Actions ---
st.sidebar.header("🔄 Actions")
if st.sidebar.button("🔍 Analyser les matchs du jour"):
    with st.spinner("Récupération des cotes et statistiques..."):
        st.session_state['analyse_faite'] = True
        st.session_state['bets'] = []
        st.session_state['all_matches'] = []
        st.session_state['combined_bets'] = []
        st.session_state['analysis_label'] = (
            "Tous les sports" if analysis_scope == "Tous les sports" else championnat
        )
        
        try:
            # Récupération des cotes (sport unique ou tous les sports)
            if analysis_scope == "Tous les sports":
                sports_to_analyze = list(sport_map.items())
            else:
                sports_to_analyze = [(championnat, sport_map[championnat])]

            matches_frames = []
            for sport_name, sport_key in sports_to_analyze:
                df_sport = get_live_odds(sport=sport_key)
                if df_sport is None or df_sport.empty:
                    continue
                df_sport = df_sport.copy()
                df_sport['sport_name'] = sport_name
                df_sport['sport_key'] = sport_key
                matches_frames.append(df_sport)

            matches = (
                pd.concat(matches_frames, ignore_index=True)
                if matches_frames
                else pd.DataFrame()
            )
            
            if matches.empty:
                st.warning("Aucun match trouvé pour les sports sélectionnés aujourd'hui.")
            else:
                coeffs = get_coefficients()
                bets = []
                all_matches_data = []
                
                for idx, row in matches.iterrows():
                    try:
                        # Récupération des stats (avec cache simple)
                        team_A_stats = get_team_stats(row['home_team'])
                        team_B_stats = get_team_stats(row['away_team'])
                        
                        # Stratégies (ici fixes, mais pourraient être dynamiques)
                        strategies_A = ["ailes", "axe"]
                        strategies_B = ["pressing_haut", "bloc_bas"]
                        
                        # Construction de la matrice et résolution
                        payoff = build_payoff_matrix(team_A_stats, team_B_stats, 
                                                     strategies_A, strategies_B, coeffs)
                        p_A, q_B, prob_home = solve_nash_2x2(payoff)
                        
                        cote_home = row['cote_home']
                        cote_draw = row.get('cote_draw', None)
                        cote_away = row.get('cote_away', None)
                        
                        match_info = {
                            'sport_name': row.get('sport_name', championnat),
                            'match': f"{row['home_team']} vs {row['away_team']}",
                            'home_team': row['home_team'],
                            'away_team': row['away_team'],
                            'match_date': row.get('match_date'),
                            'match_day': extract_match_day(row.get('match_date')),
                            'match_datetime_display': format_match_datetime(row.get('match_date')),
                            'prob_home': prob_home,
                            'prob_draw': (1 - prob_home) * 0.4,  # estimation simple
                            'prob_away': (1 - prob_home) * 0.6,
                            'cote_home': cote_home,
                            'cote_draw': cote_draw,
                            'cote_away': cote_away,
                            'prob_implicite': 1/cote_home if cote_home else 0,
                            'expected_value': None,
                            'type_paris': 'home',
                            'strategie_A': f"Freq ailes: {p_A:.0%}",
                            'strategie_B': f"Freq pressing: {q_B:.0%}",
                        }
                        
                        # Sauvegarde dans l'historique (pour obtenir un match_id)
                        match_id = ajouter_match_historique(
                            home_team=row['home_team'],
                            away_team=row['away_team'],
                            date_match=row.get('match_date', datetime.now().isoformat()),
                            xg_home=team_A_stats['xG_for'],
                            xg_away=team_B_stats['xG_for'],
                            possession_home=team_A_stats['possession'],
                            possession_away=team_B_stats['possession'],
                            strategies={"A": "ailes" if p_A >= 0.5 else "axe", "B": "pressing_haut" if q_B >= 0.5 else "bloc_bas"},
                            prob_estimee_home=prob_home,
                            cote_home=cote_home, cote_draw=cote_draw, cote_away=cote_away
                        )
                        match_info['match_id'] = match_id # Store the match_id for all matches

                        # Détection value bet
                        if cote_home:
                            kelly = compute_kelly_fraction(prob_home, cote_home, KELLY_SIMPLE)
                            kelly = min(kelly, MAX_STAKE_PCT)
                            expected_value = (prob_home * cote_home) - 1
                            match_info['expected_value'] = expected_value
                            
                            if expected_value > SEUIL_EV and kelly > 0.005: # Only add to bets if it's a value bet
                                match_info['kelly_stake'] = kelly
                                match_info['mise_conseillee'] = bankroll * kelly
                                bets.append(match_info)
                        
                        all_matches_data.append(match_info)
                        
                    except Exception as e:
                        st.error(f"Erreur sur {row['home_team']} vs {row['away_team']}: {str(e)}")
                        continue
                
                st.session_state['bets'] = bets
                st.session_state['all_matches'] = all_matches_data
                
        except Exception as e:
            st.error(f"Erreur lors de la récupération des données : {str(e)}")
            st.info("Vérifiez vos clés API dans config.py ou utilisez les données de démonstration.")

if st.sidebar.button("🔄 Recalibrer le modèle"):
    with st.spinner("Recalibrage en cours à partir de l'historique..."):
        try:
            recalibrate()
            st.sidebar.success("✅ Modèle recalibré avec succès !")
        except Exception as e:
            st.sidebar.error(f"Erreur lors du recalibrage : {e}")

# --- Zone principale ---
st.header("Dashboard des Pronostics")

# Tabs
tab1, tab2, tab3 = st.tabs(["📅 Matchs & Value Bets", "📜 Historique des Paris", "📊 Performance Globale"])
with tab1:
    current_label = st.session_state.get('analysis_label', championnat)
    st.header(f"🔍 Analyse - {current_label}")
    
    if 'analyse_faite' not in st.session_state:
        st.info("👆 Cliquez sur 'Analyser les matchs du jour' dans la barre latérale pour lancer l'analyse.")
    else:
        all_matches = st.session_state.get('all_matches', [])
        available_days = sorted(
            {
                item.get('match_day')
                for item in all_matches
                if item.get('match_day') is not None
            }
        )

        selected_days = available_days
        if available_days:
            selected_days = st.multiselect(
                "📆 Filtrer par jour",
                options=available_days,
                default=available_days,
                format_func=lambda d: d.strftime('%d/%m/%Y'),
                help="Affiche uniquement les matchs et paris des dates sélectionnées.",
            )

        selected_days_set = set(selected_days) if available_days else None

        def keep_selected_day(item):
            if selected_days_set is None:
                return True
            return item.get('match_day') in selected_days_set

        filtered_bets = [bet for bet in st.session_state.get('bets', []) if keep_selected_day(bet)]
        filtered_all_matches = [match for match in all_matches if keep_selected_day(match)]

        # --- Value Bets Section ---
        st.markdown("---")
        st.subheader("🎯 Value Bets Détectés")
        if filtered_bets:
            st.success(f"🎉 {len(filtered_bets)} Value Bet(s) prometteur(s) trouvé(s) !")
            for bet in filtered_bets:
                with st.expander(f"⚡ {bet.get('sport_name', 'N/A')} | **{bet['match']}** | EV: {bet['expected_value']:.1%}", expanded=True): # Changed title for better readability
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Date / heure", bet.get('match_datetime_display', '-'))
                        st.metric("Probabilité estimée", f"{bet['prob_home']:.1%}")
                        
                        # Récupérer et afficher le score réel si disponible
                        match_details = get_match_details(bet.get('match_id'))
                        if match_details['score']:
                            st.metric("Score réel", f"{match_details['score']} ({match_details['resultat']})")
                        st.metric("Probabilité implicite (cote)", f"{bet['prob_implicite']:.1%}")
                        st.metric("Avantage estimé", f"{bet['expected_value']:.1%}")
                    
                    with col2:
                        st.metric("Cote bookmaker", f"{bet['cote_home']:.2f}")
                        st.metric("Mise conseillée", format_fcfa(bet['mise_conseillee']))
                        st.metric("% du bankroll", f"{bet['kelly_stake']:.1%}")
                    
                    with col3:
                        st.metric("Probabilité implicite (cote)", f"{bet['prob_implicite']:.1%}") # Moved prob_implicite here
                        st.metric("Avantage estimé", f"{bet['expected_value']:.1%}") # Moved EV here
                        st.caption(f"Stratégie Domicile: {bet['strategie_A']}") # Changed label
                        st.caption(f"Stratégie Extérieur: {bet['strategie_B']}") # Changed label
                        # Bouton pour enregistrer le pari
                        if st.button(f"📝 Placer ce pari", key=f"bet_{bet['match']}"):
                            from db import enregistrer_paris
                            enregistrer_paris(
                                match=bet['match'],
                                prob_est=bet['prob_home'],
                                cote=bet['cote_home'],
                                mise=bet['mise_conseillee'], # Use the actual bet type
                                type_paris=bet['type_paris'],
                                match_id=bet.get('match_id'),
                                resultat="Pending" 
                            )
                            st.success("Pari enregistré !")
        else:
            if available_days and not selected_days:
                st.info("Aucune date sélectionnée. Choisissez au moins un jour pour afficher les résultats.")
            else:
                st.info("Aucun value bet détecté sur les jours sélectionnés.")
        
        st.markdown("---")
        st.subheader("🔗 Paris Combinés")
        if filtered_all_matches: # This block was for "Tous les matchs analysés", but it's now moved below
            df_all = pd.DataFrame(filtered_all_matches)
            # Colonnes à afficher
            cols_display = ['sport_name', 'match_datetime_display', 'match', 'prob_home', 'cote_home', 'prob_implicite', 'expected_value', 'match_id']
            for col in cols_display:
                if col not in df_all.columns:
                    df_all[col] = None
            df_display = df_all[cols_display].copy()

            # Récupérer les scores pour tous les matchs affichés
            df_display['score_reel'] = None
            df_display['resultat_reel'] = None
            for i, row in df_display.iterrows():
                if row['match_id']:
                    details = get_match_details(row['match_id'])
                    df_display.loc[i, 'score_reel'] = details['score']
                    df_display.loc[i, 'resultat_reel'] = details['resultat']

            df_display['prob_home'] = df_display['prob_home'].apply(lambda x: f"{x:.1%}")
            df_display['prob_implicite'] = df_display['prob_implicite'].apply(lambda x: f"{x:.1%}")
            df_display['expected_value'] = df_display['expected_value'].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "-"
            )

        st.markdown("---")
        st.subheader("🔗 Paris combinés")

        col_a, col_b = st.columns([2, 2])
        with col_a:
            max_matches_combo = st.radio(
                "Maximum de matchs par combiné",
                [2, 3, 4, 5],
                index=1,
                horizontal=True,
            )
        with col_b:
            if max_matches_combo == 2:
                kelly_combine_used = KELLY_COMBINE_2
            else:
                kelly_combine_used = KELLY_COMBINE_3

            st.metric(
                "Fraction Kelly (combinés)",
                f"{kelly_combine_used:.0%}",
            )

        if st.button("🔍 Générer les combinés"): # Changed button label
            singles_for_combo = filtered_bets
            if len(singles_for_combo) < 2:
                st.warning("Pas assez de paris simples rentables pour construire des combinés (minimum 2).")
                st.session_state['combined_bets'] = []
            else:
                combos = combine_bets(
                    single_bets=singles_for_combo,
                    max_matches=max_matches_combo,
                    kelly_fraction=kelly_combine_used,
                    bankroll=bankroll,
                    seuil_ev=0.0, # Keep all positive EV combos
                    max_results=3,
                )
                max_combo_stake = bankroll * MAX_STAKE_PCT * 0.5
                for combo in combos:
                    combo['mise'] = min(combo['mise'], round(max_combo_stake, 2))
                    combo['gain_potentiel'] = round(combo['mise'] * combo['cote_totale'], 2)
                st.session_state['combined_bets'] = combos

        if st.session_state.get('combined_bets'): # Check if combined_bets exist
            combos = st.session_state['combined_bets'] # Get combined bets from session state
            st.success(f"✨ {len(combos)} Combiné(s) rentable(s) trouvé(s) !")

            for idx, combo in enumerate(combos, start=1): # Loop through combined bets
                with st.expander(f"📦 Combiné {idx} | Cote: **{combo['cote_totale']:.2f}** | EV: {combo['expected_value']:.0%} | Mise: {format_fcfa(combo['mise'])}", expanded=False): # Expander for each combined bet
                    st.markdown(f"**{combo['risk_badge']} {combo['risk_label']}** : {combo['risk_description']}") # Risk profile
                    st.markdown("---")
                    for line_idx, selection in enumerate(combo['selections'], start=1): # Loop through selections in the combined bet
                        st.markdown(f"- {line_idx}. **{selection.get('sport_name', '-')}**: {selection['match']} ({selection['selection']}) | Cote: {selection['cote']:.2f} | EV: +{selection['expected_value']:.0%}") # Display each selection
                    st.markdown(f"**Gain Potentiel**: {format_fcfa(combo['gain_potentiel'])}") # Potential gain
                    # Add a button to place combined bet (requires more complex logic for DB)
                    # if st.button(f"📝 Placer ce combiné {idx}", key=f"combo_bet_{idx}"):
                    #     st.info("Fonctionnalité d'enregistrement des combinés à implémenter.")
        else:
            st.info("Aucun combiné généré. Lancez d'abord l'analyse des matchs.")

        # --- All Matches Table ---
        st.markdown("---")
        st.subheader("📋 Tous les Matchs Analysés")
        if filtered_all_matches:
            df_all = pd.DataFrame(filtered_all_matches)
            # Colonnes à afficher
            cols_display = ['sport_name', 'match_datetime_display', 'match', 'prob_home', 'cote_home', 'prob_implicite', 'expected_value', 'match_id']
            for col in cols_display:
                if col not in df_all.columns:
                    df_all[col] = None
            df_display = df_all[cols_display].copy()

            # Récupérer les scores pour tous les matchs affichés
            df_display['score_reel'] = None
            df_display['resultat_reel'] = None
            for i, row in df_display.iterrows():
                if row['match_id']:
                    details = get_match_details(row['match_id'])
                    df_display.loc[i, 'score_reel'] = details['score']
                    df_display.loc[i, 'resultat_reel'] = details['resultat']

            df_display['prob_home'] = df_display['prob_home'].apply(lambda x: f"{x:.1%}")
            df_display['prob_implicite'] = df_display['prob_implicite'].apply(lambda x: f"{x:.1%}")
            df_display['expected_value'] = df_display['expected_value'].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "-"
            )
            df_display.columns = ['Sport', 'Date / Heure', 'Match', 'Prob. Estimée', 'Cote Home', 'Prob. Implicite', 'EV', 'ID Match', 'Score Réel', 'Résultat Réel']
            
            # Apply conditional styling for EV
            def highlight_ev(s):
                if isinstance(s, str) and s.endswith('%'):
                    val = float(s.replace('%', '').replace('+', '').replace('-', ''))
                    if val > 0:
                        return 'background-color: #e6ffe6; color: green;' # Light green for positive EV
                    elif val < 0:
                        return 'background-color: #ffe6e6; color: red;' # Light red for negative EV
                return ''

            st.dataframe(df_display.style.applymap(highlight_ev, subset=['EV']), use_container_width=True)
        else:
            st.info("Aucun match à afficher pour les jours sélectionnés.")

with tab2:
    st.header("📜 Historique des paris placés")
    
    try:
        import sqlite3
        conn = sqlite3.connect(DB_NAME)
        df_paris = pd.read_sql_query("""
            SELECT match_nom AS Match, date_paris AS Date, type_paris AS Type, prob_est AS "Prob. Estimée", cote AS Cote, mise AS Mise, resultat AS Résultat, gain AS Gain 
            FROM paris 
            ORDER BY date_paris DESC 
            LIMIT 50
        """, conn)
        conn.close()
        
        if df_paris.empty:
            st.info("Aucun pari enregistré pour le moment.")
        else:
            # Formater les colonnes
            df_paris['Prob. Estimée'] = df_paris['Prob. Estimée'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
            df_paris['Date'] = pd.to_datetime(df_paris['Date']).dt.strftime('%d/%m/%Y %H:%M')
            # df_paris['Type'] = df_paris['Type'].apply(lambda x: x.capitalize()) # Optional: capitalize bet type
            df_paris['mise'] = df_paris['mise'].apply(format_fcfa)
            df_paris['gain'] = df_paris['gain'].apply(format_fcfa)
            
            # Colorer les résultats
            def color_result(val):
                if val == 'Win':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'Loss':
                    return 'background-color: #f8d7da; color: #721c24'
                return ''
            st.dataframe(df_paris.style.applymap(color_result, subset=['Résultat']), use_container_width=True)
            
    except Exception as e:
        st.error(f"Erreur lors de la lecture de l'historique : {e}")

with tab3:
    st.header("📊 Performance du système")
    
    try:
        import sqlite3
        conn = sqlite3.connect(DB_NAME)
        df_perf = pd.read_sql_query("""
            SELECT * FROM paris WHERE resultat != 'Pending'
        """, conn)
        conn.close()
        
        if df_perf.empty:
            st.info("Pas encore assez de paris terminés pour afficher les statistiques de performance.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            nb_paris = len(df_perf)
            nb_wins = len(df_perf[df_perf['resultat'] == 'Win'])
            win_rate = nb_wins / nb_paris * 100 if nb_paris > 0 else 0

            df_perf['profit'] = df_perf['gain'] - df_perf['mise']
            profit_total = df_perf['profit'].sum()
            roi = profit_total / df_perf['mise'].sum() * 100 if df_perf['mise'].sum() > 0 else 0
            
            with col1:
                st.metric("Total Paris", nb_paris)
            with col2:
                st.metric("Taux de Réussite", f"{win_rate:.1f}%")
            with col3:
                st.metric("Profit Net", format_fcfa(profit_total))
            with col4:
                st.metric("ROI", f"{roi:.1f}%")
            
            st.markdown("---")
            # Graphique d'évolution du bankroll
            st.subheader("📈 Évolution du Bankroll")
            if nb_paris > 1:
                st.subheader("📈 Évolution du bankroll")
                df_perf = df_perf.sort_values('date_paris')
                df_perf['bankroll'] = bankroll + df_perf['profit'].cumsum()
                st.line_chart(df_perf.set_index(pd.to_datetime(df_perf['date_paris']))['bankroll'])
            else:
                st.info("Plus de paris sont nécessaires pour afficher l'évolution du bankroll.")
            
            # Distribution des cotes
            st.subheader("📊 Distribution des cotes jouées")
            fig = pd.DataFrame({
                'Résultat': ['Win' if r == 'Win' else 'Loss' for r in df_perf['resultat']],
                'Cote': df_perf['cote']
            })
            st.scatter_chart(fig, x='Cote', y='Résultat')
            
    except Exception as e:
        st.error(f"Erreur lors du calcul des performances : {e}")

# --- Pied de page ---
st.markdown("---")
st.caption("""
⚠️ **Avertissement** : Ce système est un outil d'aide à la décision basé sur la théorie des jeux.
Il ne garantit pas de gains. Les paris sportifs comportent des risques. Ne jouez que ce que vous pouvez vous permettre de perdre.
""")

# --- Actualisation automatique (optionnelle) ---
if st.checkbox("🔄 Actualisation automatique (toutes les heures)"):
        if MODE == "production": # Check if MODE is defined and equals "production"
                st.success("Actualisation automatique activée : la page se recharge toutes les heures.")
                st.components.v1.html(
                        """
                        <script>
                            setTimeout(function() {
                                window.location.reload();
                            }, 3600000);
                        </script>
                        """,
                        height=0,
                )
        else:
                st.info("L'actualisation automatique n'est pas active en local. Lancez le script main.py avec schedule.")