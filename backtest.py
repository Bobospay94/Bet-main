import pandas as pd
import numpy as np
from matrix_builder import build_payoff_matrix, estimer_stats_depuis_cotes
from nash_equilibrium import solve_nash_2x2
from value_detector import calculate_expected_value, calculate_kelly_fraction

def run_backtest(data_file):
    """
    Simule les performances du modèle sur des données historiques.
    Le CSV doit contenir : cote_home, cote_draw, cote_away, resultat (home/draw/away)
    """
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"❌ Fichier {data_file} introuvable.")
        return

    initial_capital = 1000.0
    capital = initial_capital
    history = []

    # Paramètres de simulation
    KELLY_FRAC = 0.25  # Quart de Kelly
    
    print(f"🚀 Démarrage du backtest sur {len(df)} matchs...")

    for idx, row in df.iterrows():
        # 1. Estimer les probabilités via la Théorie des Jeux
        stats_h, stats_a = estimer_stats_depuis_cotes(row['cote_home'], row['cote_draw'], row['cote_away'])
        
        # On utilise les stratégies par défaut pour reconstruire la matrice
        payoff_matrix = build_payoff_matrix(stats_h, stats_a, ["ailes", "axe"], ["pressing_haut", "bloc_bas"])
        
        # Résolution de Nash pour obtenir la probabilité "réelle" estimée
        _, _, prob_home = solve_nash_2x2(payoff_matrix)
        
        # 2. Détection de Value
        ev = calculate_expected_value(prob_home, row['cote_home'])
        
        if ev > 0.02:  # Seuil de 2% de value
            # 3. Calcul de la mise via Kelly
            fraction = calculate_kelly_fraction(prob_home, row['cote_home'], fraction=KELLY_FRAC)
            mise = capital * fraction
            
            if mise > 0:
                # 4. Résultat du pari
                won = (row['resultat'] == 'home')
                gain = (mise * row['cote_home']) if won else 0
                capital = capital - mise + gain
                
                history.append({
                    'match': idx,
                    'ev': ev,
                    'mise': mise,
                    'win': won,
                    'capital': capital
                })

    roi = ((capital - initial_capital) / initial_capital) * 100
    print(f"\n--- RÉSULTATS DU BACKTEST ---")
    print(f"Capital Initial : {initial_capital:.2f}")
    print(f"Capital Final   : {capital:.2f}")
    print(f"ROI Global      : {roi:.2f}%")
    print(f"Nombre de paris : {len(history)}")
    print(f"Win Rate        : {sum(1 for x in history if x['win']) / len(history):.1%}" if history else "N/A")

if __name__ == "__main__":
    run_backtest('historique_matchs.csv')