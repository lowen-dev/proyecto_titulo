import numpy as np
from mle import F, g

# ==========================================================
# MÍNIMOS CUADRADOS ORDINARIOS (MCO / OLS), caso mono-paramétrico
# ==========================================================
# Se minimiza  S = sum_i (D_i - F_i)^2   (sin pesos)
# con F_i = mu + A * g_i. Igual que en mle.py, se estima un
# parámetro dejando el otro fijo en su valor real.
#
# Como el modelo es lineal en cada parámetro, la solución es cerrada:
#   dS/dmu = 0  ->  mu_hat = mean(D - A0 * g)
#   dS/dA  = 0  ->  A_hat  = sum(g * (D - mu0)) / sum(g^2)
# No hace falta optimizador numérico.


def estimar_mco_individual(D, x, c, sigma, mu_0, A_0):
    """Estimación MCO para una sola muestra D (shape: (len(x),))."""
    g_vals = g(x, c, sigma)

    # MCO de mu
    mu_mco = np.mean(D - A_0 * g_vals)

    # MCO de 
    A_mco = np.sum(g_vals * (D - mu_0)) / np.sum(g_vals**2)

    return mu_mco, A_mco


def mco(n_muestras, x, c, sigma, mu_rng, A_rng, D_total=None):
    """
    Aplica MCO a n_muestras espectros.
    Si se pasa D_total (el mismo que usó el MLE), se reutiliza para
    comparar ambos estimadores sobre exactamente los mismos datos.
    """
    if D_total is None:
        D_total = np.random.poisson(
            lam=F(mu_rng, A_rng, x, c, sigma),
            size=(n_muestras, len(x))
        )

    g_vals = g(x, c, sigma)

    # Versión vectorizada: aplica la fórmula cerrada a todas las filas a la vez
    mu_resultados = np.mean(D_total - A_rng * g_vals, axis=1)
    A_resultados = np.sum(g_vals * (D_total - mu_rng), axis=1) / np.sum(g_vals**2)

    return mu_resultados, A_resultados, D_total