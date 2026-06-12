"""
Nash Equilibrium - Résolution du jeu à somme nulle
====================================================

Ce module calcule l'équilibre de Nash en stratégies mixtes pour un jeu 2×2.
Le jeu modélise l'affrontement tactique entre deux équipes :
- L'équipe A choisit entre deux stratégies offensives
- L'équipe B choisit entre deux stratégies défensives
- Le gain est la probabilité que l'équipe A gagne le match

Mathématiquement, on résout :
    p * a11 + (1-p) * a21 = p * a12 + (1-p) * a22  (indifférence de B)
    q * a11 + (1-q) * a12 = q * a21 + (1-q) * a22  (indifférence de A)

où p = probabilité que A joue la stratégie 1
      q = probabilité que B joue la stratégie 1
      aij = gain pour A quand A joue i et B joue j
"""

import numpy as np
import math
from scipy.optimize import linprog
import warnings


# ============================================================
# 1. RÉSOLUTION ANALYTIQUE DU JEU 2×2
# ============================================================

def solve_nash_2x2(payoff_A, tolerance=1e-10):
    """
    Résout l'équilibre de Nash en stratégies mixtes pour un jeu 2×2 à somme nulle.
    
    Formules analytiques :
    p = (a22 - a21) / (a11 - a12 - a21 + a22)
    q = (a22 - a12) / (a11 - a12 - a21 + a22)
    v = (a11*a22 - a12*a21) / (a11 - a12 - a21 + a22)
    
    Args:
        payoff_A: numpy.array 2×2 des gains pour le joueur A
                  [[a11, a12],
                   [a21, a22]]
        tolerance: seuil pour considérer le dénominateur comme nul
    
    Returns:
        tuple: (p, q, v) où :
            - p : probabilité que A joue la stratégie 1 (entre 0 et 1)
            - q : probabilité que B joue la stratégie 1 (entre 0 et 1)
            - v : valeur du jeu (gain espéré pour A à l'équilibre)
    
    Raises:
        ValueError: Si la matrice ne permet pas un équilibre mixte unique
    """
    
    a11, a12 = payoff_A[0, 0], payoff_A[0, 1]
    a21, a22 = payoff_A[1, 0], payoff_A[1, 1]
    
    # Dénominateur commun
    denom = a11 - a12 - a21 + a22
    
    # Cas 1 : Équilibre mixte unique (denom ≠ 0)
    if abs(denom) > tolerance:
        # Probabilité que A joue la stratégie 1
        p = (a22 - a21) / denom
        # Probabilité que B joue la stratégie 1
        q = (a22 - a12) / denom
        
        # Valeur du jeu (espérance pour A)
        v = (a11 * a22 - a12 * a21) / denom
        
        # Borner entre 0 et 1
        p = max(0.0, min(1.0, p))
        q = max(0.0, min(1.0, q))
        
        return p, q, v
    
    # Cas 2 : Dénominateur nul → stratégies pures ou infinité d'équilibres
    else:
        return _solve_degenerate_case(payoff_A, tolerance)


def _solve_degenerate_case(payoff_A, tolerance=1e-10):
    """
    Résout les cas dégénérés où le dénominateur est nul.
    Cela arrive quand les lignes ou les colonnes sont identiques,
    ou quand une stratégie pure est dominante.
    """
    
    a11, a12 = payoff_A[0, 0], payoff_A[0, 1]
    a21, a22 = payoff_A[1, 0], payoff_A[1, 1]
    
    # Vérifier si A a une stratégie dominante
    if a11 >= a21 and a12 >= a22:
        # La stratégie 1 de A domine faiblement la stratégie 2
        if a11 > a21 or a12 > a22:
            # Domination stricte : A joue stratégie 1 pure
            p = 1.0
            # B choisit la meilleure réponse
            if a11 < a12:
                q, v = 0.0, a12
            else:
                q, v = 1.0, a11
            return p, q, v
    
    if a21 >= a11 and a22 >= a12:
        # La stratégie 2 de A domine
        if a21 > a11 or a22 > a12:
            p = 0.0
            if a21 < a22:
                q, v = 0.0, a22
            else:
                q, v = 1.0, a21
            return p, q, v
    
    # Vérifier si B a une stratégie dominante (rappel : B veut minimiser le gain de A)
    if a11 <= a12 and a21 <= a22:
        # La stratégie 1 de B domine (car elle donne un gain plus faible à A)
        if a11 < a12 or a21 < a22:
            q = 1.0
            if a11 > a21:
                p, v = 0.0, a21
            else:
                p, v = 1.0, a11
            return p, q, v
    
    if a12 <= a11 and a22 <= a21:
        if a12 < a11 or a22 < a21:
            q = 0.0
            if a12 > a22:
                p, v = 0.0, a22
            else:
                p, v = 1.0, a12
            return p, q, v
    
    # Cas d'égalité partout : tout est équilibre
    # On prend le milieu par convention
    if abs(a11 - a12) < tolerance and abs(a11 - a21) < tolerance and abs(a11 - a22) < tolerance:
        return 0.5, 0.5, a11
    
    # Cas résiduel : on utilise la programmation linéaire
    return _solve_with_linear_programming(payoff_A)


# ============================================================
# 2. RÉSOLUTION PAR PROGRAMMATION LINÉAIRE (méthode générale)
# ============================================================

def _solve_with_linear_programming(payoff_A):
    """
    Résout le jeu à somme nulle par programmation linéaire.
    Méthode générale qui fonctionne même pour les cas dégénérés.
    
    Pour le joueur A (maximisateur) :
        max v
        s.c. p1*a11 + p2*a21 >= v
             p1*a12 + p2*a22 >= v
             p1 + p2 = 1
             p1, p2 >= 0
    
    On résout le dual pour plus de stabilité numérique.
    """
    
    a11, a12 = payoff_A[0, 0], payoff_A[0, 1]
    a21, a22 = payoff_A[1, 0], payoff_A[1, 1]
    
    # On résout pour le joueur B (plus simple en PL standard)
    # min v
    # s.c. q1*a11 + q2*a12 <= v
    #      q1*a21 + q2*a22 <= v
    #      q1 + q2 = 1
    #      q1, q2 >= 0
    
    # Reformulation : on pose x1 = q1/v, x2 = q2/v
    # max x1 + x2
    # s.c. a11*x1 + a12*x2 <= 1
    #      a21*x1 + a22*x2 <= 1
    #      x1, x2 >= 0
    # puis v = 1/(x1+x2), q1 = x1*v, q2 = x2*v
    
    # Coefficients de la fonction objectif (on maximise x1 + x2, donc on minimise -x1 - x2)
    c = [-1, -1]
    
    # Contraintes d'inégalité : A_ub @ x <= b_ub
    A_ub = [[a11, a12],
            [a21, a22]]
    b_ub = [1, 1]
    
    # Bornes : x1, x2 >= 0
    bounds = [(0, None), (0, None)]
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x1, x2 = result.x
            if x1 + x2 > 1e-10:
                v = 1 / (x1 + x2)
                q1 = x1 * v
                q2 = x2 * v
                
                # Récupérer p via le dual
                # Les variables duales donnent la stratégie de A
                if hasattr(result, 'ineqlin') and result.ineqlin is not None:
                    p1 = result.ineqlin.marginals[0] if result.ineqlin.marginals[0] > 0 else 0.5
                    p2 = result.ineqlin.marginals[1] if result.ineqlin.marginals[1] > 0 else 0.5
                    somme = p1 + p2
                    if somme > 1e-10:
                        p1 /= somme
                        p2 /= somme
                    else:
                        p1, p2 = 0.5, 0.5
                else:
                    p1, p2 = 0.5, 0.5
                
                return p1, q1, v
            
    except Exception:
        pass
    
    # Fallback ultime
    return 0.5, 0.5, (a11 + a12 + a21 + a22) / 4


# ============================================================
# 3. FONCTIONS AVANCÉES POUR JEUX N×M
# ============================================================

def solve_nash_general(payoff_A):
    """
    Résout l'équilibre de Nash pour un jeu à somme nulle de taille quelconque.
    Utilise la programmation linéaire.
    
    Args:
        payoff_A: numpy.array n×m des gains pour A
    
    Returns:
        tuple: (strategie_A, strategie_B, valeur) où :
            - strategie_A: array des probabilités pour A
            - strategie_B: array des probabilités pour B
            - valeur: valeur du jeu
    """
    
    n, m = payoff_A.shape
    
    # Résolution pour B (comme dans le cas 2×2 mais généralisé)
    # max sum(x_j)
    # s.c. sum_j(a_ij * x_j) <= 1 pour tout i
    #      x_j >= 0
    
    c = [-1] * m  # On minimise -sum(x_j)
    
    A_ub = payoff_A.T.tolist()  # Transposé : chaque ligne = une contrainte
    b_ub = [1] * n
    
    bounds = [(0, None)] * m
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
            somme_x = sum(x)
            
            if somme_x > 1e-10:
                v = 1 / somme_x
                strategie_B = np.array([xj * v for xj in x])
                
                # Récupérer la stratégie de A via le dual
                if hasattr(result, 'ineqlin') and result.ineqlin is not None:
                    marginals = result.ineqlin.marginals
                    somme_marg = sum(marginals)
                    if somme_marg > 1e-10:
                        strategie_A = np.array([m / somme_marg for m in marginals])
                    else:
                        strategie_A = np.ones(n) / n
                else:
                    strategie_A = np.ones(n) / n
                
                return strategie_A, strategie_B, v
                
    except Exception as e:
        print(f"Erreur PL : {e}")
    
    # Fallback
    return np.ones(n) / n, np.ones(m) / m, 0.5


# ============================================================
# 4. ANALYSE DE SENSIBILITÉ
# ============================================================

def analyze_sensitivity(payoff_A, delta=0.01):
    """
    Analyse la sensibilité de l'équilibre à de petites variations.
    Utile pour comprendre la robustesse des stratégies.
    
    Args:
        payoff_A: matrice 2×2
        delta: variation à appliquer
    
    Returns:
        dict avec les dérivées partielles approximées
    """
    
    p_base, q_base, v_base = solve_nash_2x2(payoff_A)
    
    sensibilites = {}
    
    # Sensibilité de p par rapport à a11
    payoff_pert = payoff_A.copy()
    payoff_pert[0, 0] += delta
    p_pert, _, _ = solve_nash_2x2(payoff_pert)
    sensibilites['dp_da11'] = (p_pert - p_base) / delta
    
    # Sensibilité de p par rapport à a12
    payoff_pert = payoff_A.copy()
    payoff_pert[0, 1] += delta
    p_pert, _, _ = solve_nash_2x2(payoff_pert)
    sensibilites['dp_da12'] = (p_pert - p_base) / delta
    
    # Sensibilité de v par rapport à chaque entrée
    for i in range(2):
        for j in range(2):
            payoff_pert = payoff_A.copy()
            payoff_pert[i, j] += delta
            _, _, v_pert = solve_nash_2x2(payoff_pert)
            sensibilites[f'dv_da{i+1}{j+1}'] = (v_pert - v_base) / delta
    
    return sensibilites


# ============================================================
# 5. INTERPRÉTATION ET AFFICHAGE
# ============================================================

def interpret_equilibrium(p, q, v, strategies_A, strategies_B, 
                          equipe_A="Équipe A", equipe_B="Équipe B"):
    """
    Fournit une interprétation en langage naturel de l'équilibre trouvé.
    
    Args:
        p, q, v: résultats de solve_nash_2x2
        strategies_A: liste [nom_strat1, nom_strat2]
        strategies_B: liste [nom_strat1, nom_strat2]
        equipe_A, equipe_B: noms des équipes
    
    Returns:
        str: Interprétation textuelle
    """
    lignes = []
    lignes.append(f"\n{'='*60}")
    lignes.append(f"  INTERPRÉTATION DE L'ÉQUILIBRE DE NASH")
    lignes.append(f"{'='*60}\n")
    
    # Stratégie de A
    if p > 0.95:
        lignes.append(f"🔒 {equipe_A} doit jouer **{strategies_A[0]}** presque tout le temps ({p:.1%}).")
    elif p < 0.05:
        lignes.append(f"🔒 {equipe_A} doit jouer **{strategies_A[1]}** presque tout le temps ({1-p:.1%}).")
    elif 0.45 <= p <= 0.55:
        lignes.append(f"🎲 {equipe_A} doit alterner de façon équilibrée entre **{strategies_A[0]}** ({p:.1%}) et **{strategies_A[1]}** ({1-p:.1%}).")
    else:
        dominante = strategies_A[0] if p > 0.5 else strategies_A[1]
        freq = max(p, 1-p)
        lignes.append(f"📊 {equipe_A} doit privilégier **{dominante}** ({freq:.1%}) tout en variant avec l'autre stratégie ({1-freq:.1%}).")
    
    # Stratégie de B
    if q > 0.95:
        lignes.append(f"🔒 {equipe_B} doit jouer **{strategies_B[0]}** presque tout le temps ({q:.1%}).")
    elif q < 0.05:
        lignes.append(f"🔒 {equipe_B} doit jouer **{strategies_B[1]}** presque tout le temps ({1-q:.1%}).")
    elif 0.45 <= q <= 0.55:
        lignes.append(f"🎲 {equipe_B} doit alterner de façon équilibrée entre **{strategies_B[0]}** ({q:.1%}) et **{strategies_B[1]}** ({1-q:.1%}).")
    else:
        dominante = strategies_B[0] if q > 0.5 else strategies_B[1]
        freq = max(q, 1-q)
        lignes.append(f"📊 {equipe_B} doit privilégier **{dominante}** ({freq:.1%}) tout en variant avec l'autre stratégie ({1-freq:.1%}).")
    
    # Valeur du jeu
    lignes.append(f"\n📈 À l'équilibre, **{equipe_A}** a une probabilité de victoire estimée à **{v:.1%}**.")
    
    if v > 0.65:
        lignes.append(f"   → {equipe_A} est clairement favori, quelle que soit la stratégie adverse.")
    elif v > 0.55:
        lignes.append(f"   → {equipe_A} a un avantage modéré mais significatif.")
    elif v > 0.45:
        lignes.append(f"   → Le match est équilibré, avec un léger avantage pour {'A' if v >= 0.5 else 'B'}.")
    elif v > 0.35:
        lignes.append(f"   → {equipe_B} a un avantage modéré.")
    else:
        lignes.append(f"   → {equipe_B} est clairement favori.")
    
    lignes.append(f"\n{'='*60}\n")
    
    return "\n".join(lignes)


def format_strategy(p, q, v, strategies_A, strategies_B):
    """
    Retourne un dictionnaire formaté pour affichage Streamlit.
    """
    return {
        'strategie_A_1': strategies_A[0],
        'freq_A_1': round(p * 100, 1),
        'strategie_A_2': strategies_A[1],
        'freq_A_2': round((1-p) * 100, 1),
        'strategie_B_1': strategies_B[0],
        'freq_B_1': round(q * 100, 1),
        'strategie_B_2': strategies_B[1],
        'freq_B_2': round((1-q) * 100, 1),
        'prob_victoire_A': round(v * 100, 1),
        'prob_victoire_B': round((1-v) * 100, 1)
    }


# ============================================================
# 6. VALIDATION
# ============================================================

def verify_equilibrium(payoff_A, p, q, v, tolerance=1e-6):
    """
    Vérifie que (p, q, v) est bien un équilibre de Nash.
    
    Vérifications :
    1. Aucun joueur ne peut améliorer son gain en déviant unilatéralement
    2. La valeur correspond bien à l'espérance
    
    Returns:
        dict avec les résultats des vérifications
    """
    
    a11, a12 = payoff_A[0, 0], payoff_A[0, 1]
    a21, a22 = payoff_A[1, 0], payoff_A[1, 1]
    
    # Gain espéré pour A à l'équilibre
    expected_A = p * q * a11 + p * (1-q) * a12 + (1-p) * q * a21 + (1-p) * (1-q) * a22
    
    # Gain si A dévie vers stratégie 1 pure
    gain_A1 = q * a11 + (1-q) * a12
    # Gain si A dévie vers stratégie 2 pure
    gain_A2 = q * a21 + (1-q) * a22
    
    # Gain si B dévie vers stratégie 1 pure (B minimise)
    gain_B1 = p * a11 + (1-p) * a21
    # Gain si B dévie vers stratégie 2 pure
    gain_B2 = p * a12 + (1-p) * a22
    
    checks = {
        'valeur_correcte': abs(expected_A - v) < tolerance,
        'A_pas_incite_devier': gain_A1 <= v + tolerance and gain_A2 <= v + tolerance,
        'B_pas_incite_devier': gain_B1 >= v - tolerance and gain_B2 >= v - tolerance,
        'ecart_valeur': round(expected_A - v, 10),
        'gain_A1': round(gain_A1, 6),
        'gain_A2': round(gain_A2, 6),
        'gain_B1': round(gain_B1, 6),
        'gain_B2': round(gain_B2, 6)
    }
    
    checks['equilibre_valide'] = (checks['valeur_correcte'] and 
                                   checks['A_pas_incite_devier'] and 
                                   checks['B_pas_incite_devier'])
    
    return checks


# ============================================================
# 7. TEST
# ============================================================

if __name__ == "__main__":
    print("🧪 Test du module nash_equilibrium\n")
    
    strategies_A = ["ailes", "axe"]
    strategies_B = ["pressing_haut", "bloc_bas"]
    
    # Test 1 : Cas standard
    print("=" * 60)
    print("TEST 1 : Cas standard")
    print("=" * 60)
    mat1 = np.array([[0.50, 0.80],
                     [0.90, 0.60]])
    p1, q1, v1 = solve_nash_2x2(mat1)
    print(f"Matrice :\n{mat1}")
    print(f"p = {p1:.3f}, q = {q1:.3f}, v = {v1:.3f}")
    print(interpret_equilibrium(p1, q1, v1, strategies_A, strategies_B))
    
    # Vérification
    verif1 = verify_equilibrium(mat1, p1, q1, v1)
    print(f"Équilibre valide : {verif1['equilibre_valide']}")
    
    # Test 2 : Stratégie dominante
    print("=" * 60)
    print("TEST 2 : Stratégie dominante")
    print("=" * 60)
    mat2 = np.array([[0.80, 0.90],
                     [0.30, 0.40]])
    p2, q2, v2 = solve_nash_2x2(mat2)
    print(f"Matrice :\n{mat2}")
    print(f"p = {p2:.3f}, q = {q2:.3f}, v = {v2:.3f}")
    print(interpret_equilibrium(p2, q2, v2, strategies_A, strategies_B))
    
    # Test 3 : Équilibre 50/50
    print("=" * 60)
    print("TEST 3 : Équilibre parfaitement symétrique")
    print("=" * 60)
    mat3 = np.array([[0.70, 0.30],
                     [0.30, 0.70]])
    p3, q3, v3 = solve_nash_2x2(mat3)
    print(f"Matrice :\n{mat3}")
    print(f"p = {p3:.3f}, q = {q3:.3f}, v = {v3:.3f}")
    print(interpret_equilibrium(p3, q3, v3, strategies_A, strategies_B))
    
    # Test 4 : Cas dégénéré
    print("=" * 60)
    print("TEST 4 : Cas dégénéré (lignes identiques)")
    print("=" * 60)
    mat4 = np.array([[0.60, 0.80],
                     [0.60, 0.80]])
    try:
        p4, q4, v4 = solve_nash_2x2(mat4)
        print(f"Matrice :\n{mat4}")
        print(f"p = {p4:.3f}, q = {q4:.3f}, v = {v4:.3f}")
    except Exception as e:
        print(f"Erreur : {e}")
    
    # Test 5 : Analyse de sensibilité
    print("=" * 60)
    print("TEST 5 : Analyse de sensibilité")
    print("=" * 60)
    sensibilites = analyze_sensitivity(mat1, delta=0.01)
    for k, v in sensibilites.items():
        print(f"  {k} = {v:.4f}")
    
    # Test 6 : Jeu 3×3
    print("=" * 60)
    print("TEST 6 : Jeu 3×3")
    print("=" * 60)
    mat5 = np.array([[0.50, 0.60, 0.70],
                     [0.40, 0.50, 0.60],
                     [0.30, 0.40, 0.50]])
    strat_A, strat_B, val = solve_nash_general(mat5)
    print(f"Matrice :\n{mat5}")
    print(f"Stratégie A : {[f'{p:.3f}' for p in strat_A]}")
    print(f"Stratégie B : {[f'{p:.3f}' for p in strat_B]}")
    print(f"Valeur : {val:.3f}")
    
    # Test 7 : Performance
    print("\n" + "=" * 60)
    print("TEST 7 : Performance (1000 résolutions)")
    print("=" * 60)
    import time
    start = time.time()
    for _ in range(10000):
        solve_nash_2x2(mat1)
    elapsed = time.time() - start
    print(f"10 000 résolutions en {elapsed:.3f}s ({elapsed/10000*1000:.3f}ms/résolution)")
    
    print("\n✅ Tests terminés !")