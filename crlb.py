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
        "cumple_mu": cumple_mu,
        "var_A_mle": var_A_mle,
        "var_crlb_A": var_crlb_A,
        "cumple_A": cumple_A,
    }
    
    return resultados

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