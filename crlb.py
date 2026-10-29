import numpy as np
from mle import F, g

# ==========================================
# COTA DE CRAMÉR-RAO 1D (Mono-paramétrica)
# ==========================================

def crlb_mu(mu, A, x, c, sigma):
    """
    Cota Inferior de Cramér-Rao 1D para mu 
    (asumiendo que 'A' es conocido y fijo).
    Var(mu) >= 1 / I_11 = 1 / sum( 1 / F(x) )
    """
    F_vals = F(mu, A, x, c, sigma)
    I_11 = np.sum(1.0 / F_vals)
    return 1.0 / I_11

def crlb_A(mu, A, x, c, sigma):
    """
    Cota Inferior de Cramér-Rao 1D para A 
    (asumiendo que 'mu' es conocido y fijo).
    Var(A) >= 1 / I_22 = 1 / sum( g(x)^2 / F(x) )
    """
    F_vals = F(mu, A, x, c, sigma)
    g_vals = g(x, c, sigma)
    I_22 = np.sum((g_vals**2) / F_vals)
    return 1.0 / I_22

# ==========================================
# VARIANZA TEÓRICA DE MCO BAJO RUIDO DE POISSON
# ==========================================
# Con Var(D_i) = F_i (heterocedástico), la varianza exacta del
# estimador MCO es:
#   Var(mu_hat) = sum(F_i) / N^2
#   Var(A_hat)  = sum(g_i^2 * F_i) / (sum g_i^2)^2
# Por Cauchy-Schwarz estas son >= CRLB, con igualdad solo si F es constante.

def var_teorica_mco_mu(mu, A, x, c, sigma):
    """Varianza teórica del estimador MCO de mu (A conocido)."""
    F_vals = F(mu, A, x, c, sigma)
    return np.sum(F_vals) / len(x)**2

def var_teorica_mco_A(mu, A, x, c, sigma):
    """Varianza teórica del estimador MCO de A (mu conocido)."""
    F_vals = F(mu, A, x, c, sigma)
    g_vals = g(x, c, sigma)
    return np.sum(g_vals**2 * F_vals) / np.sum(g_vals**2)**2

# ==========================================
# COMPARACIÓN CON RESULTADOS MLE
# ==========================================

def evaluar_crlb(mu_array, A_array, mu_real, A_real, x, c, sigma):
    """
    Compara las varianzas empíricas del Monte Carlo con las cotas CRLB 1D teóricas.
    """
    # 1. Varianzas empíricas de las simulaciones MLE
    var_mu_mle = np.var(mu_array, ddof=1)
    var_A_mle = np.var(A_array, ddof=1)

    # 2. Cotas teóricas CRLB 1D evaluadas en los valores reales
    var_crlb_mu = crlb_mu(mu_real, A_real, x, c, sigma)
    var_crlb_A = crlb_A(mu_real, A_real, x, c, sigma)

    # 3. CRLB
    N = len(mu_array)
    tol = 3 * np.sqrt(2.0 / (N - 1))      # ~3 sigmas de la fluctuación de la varianza muestral

    ratio_mu = var_mu_mle / var_crlb_mu
    ratio_A  = var_A_mle  / var_crlb_A

    cumple_mu = "SÍ" if ratio_mu >= 1 - tol else "NO"
    cumple_A  = "SÍ" if ratio_A  >= 1 - tol else "NO"

    resultados = {
        "var_mu_mle": var_mu_mle,
        "var_crlb_mu": var_crlb_mu,
        "ratio_mu": ratio_mu,
        "cumple_mu": cumple_mu,
        "var_A_mle": var_A_mle,
        "var_crlb_A": var_crlb_A,
        "ratio_A": ratio_A,
        "cumple_A": cumple_A,
    }
    
    return resultados

def evaluar_mco(mu_array, A_array, mu_real, A_real, x, c, sigma):
    """
    Igual que evaluar_crlb pero para las simulaciones MCO. Además de la CRLB,
    compara con la varianza teórica de MCO (chequeo de la simulación).
    Mismo criterio: ratio >= 1 - tol.
    """
    var_mu_mco = np.var(mu_array, ddof=1)
    var_A_mco = np.var(A_array, ddof=1)

    var_crlb_mu = crlb_mu(mu_real, A_real, x, c, sigma)
    var_crlb_A = crlb_A(mu_real, A_real, x, c, sigma)

    var_teo_mu = var_teorica_mco_mu(mu_real, A_real, x, c, sigma)
    var_teo_A = var_teorica_mco_A(mu_real, A_real, x, c, sigma)

    N = len(mu_array)
    tol = 3 * np.sqrt(2.0 / (N - 1))

    ratio_mu = var_mu_mco / var_crlb_mu
    ratio_A  = var_A_mco  / var_crlb_A

    return {
        "var_mu_mco": var_mu_mco,
        "var_teo_mu": var_teo_mu,
        "var_crlb_mu": var_crlb_mu,
        "ratio_mu": ratio_mu,
        "cumple_mu": "SÍ" if ratio_mu >= 1 - tol else "NO",
        "var_A_mco": var_A_mco,
        "var_teo_A": var_teo_A,
        "var_crlb_A": var_crlb_A,
        "ratio_A": ratio_A,
        "cumple_A": "SÍ" if ratio_A >= 1 - tol else "NO",
    }

def imprimir_reporte_1d(res):
    """Muestra en consola el reporte con varianzas y validación de la CRLB."""
    print("\n" + "="*78)
    print("      ANÁLISIS DE EFICIENCIA: MLE MONO-PARAMÉTRICO vs. CRLB 1D")
    print("="*78)
    print(f"{'Parámetro':<12} | {'Varianza MLE (Empírica)':<22} | {'Varianza CRLB (Teórica)':<25} | {'¿Cumple CRLB?':<12}")
    print("-" * 78)
    print(f"{'mu':<12} | {res['var_mu_mle']:<22.4f} | {res['var_crlb_mu']:<25.4f} | {res['cumple_mu']:<12}")
    print(f"{'A':<12} | {res['var_A_mle']:<22.4f} | {res['var_crlb_A']:<25.4f} | {res['cumple_A']:<12}")
    print("="*78 + "\n")

def imprimir_reporte_comparativo(res_mle, res_mco):
    """Análisis de eficiencia conjunto: MLE y MCO frente a la CRLB 1D."""
    w = 96
    print("\n" + "="*w)
    print("      ANÁLISIS DE EFICIENCIA: MLE vs. MCO FRENTE A LA CRLB 1D")
    print("="*w)
    print(f"{'Param.':<7} | {'Estimador':<9} | {'Var empírica':<13} | {'Var CRLB':<11} | "
          f"{'Var/CRLB':<9} | {'¿Cumple CRLB?':<13} | {'Var teórica':<11}")
    print("-"*w)
    filas = [
        ("mu", "MLE", res_mle['var_mu_mle'], res_mle['var_crlb_mu'], res_mle['ratio_mu'], res_mle['cumple_mu'], None),
        ("mu", "MCO", res_mco['var_mu_mco'], res_mco['var_crlb_mu'], res_mco['ratio_mu'], res_mco['cumple_mu'], res_mco['var_teo_mu']),
        ("A",  "MLE", res_mle['var_A_mle'],  res_mle['var_crlb_A'],  res_mle['ratio_A'],  res_mle['cumple_A'],  None),
        ("A",  "MCO", res_mco['var_A_mco'],  res_mco['var_crlb_A'],  res_mco['ratio_A'],  res_mco['cumple_A'],  res_mco['var_teo_A']),
    ]
    for p, e, v, cr, r, ok, teo in filas:
        teo_s = f"{teo:<11.4f}" if teo is not None else f"{'-':<11}"
        print(f"{p:<7} | {e:<9} | {v:<13.4f} | {cr:<11.4f} | {r:<9.4f} | {ok:<13} | {teo_s}")
        if e == "MCO":
            print("-"*w)
    print("="*w)
    # Eficiencia relativa: Var(MLE) / Var(MCO)  (<1 => MLE más eficiente)
    er_mu = res_mle['var_mu_mle'] / res_mco['var_mu_mco']
    er_A  = res_mle['var_A_mle']  / res_mco['var_A_mco']
    print(f"Eficiencia relativa Var(MLE)/Var(MCO):  mu = {er_mu:.4f}   |   A = {er_A:.4f}   (< 1 => MLE más preciso)")
    print("="*w + "\n")