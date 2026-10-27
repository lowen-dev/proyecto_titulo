import numpy as np
from scipy.optimize import brentq


def g(x, c, sigma):
    """Función base g(x) de Irland."""
    return np.exp(-(x - c)**2 / (2 * sigma**2))

def F(mu, A, x, c, sigma):
    """Modelo F(x) = mu + A * g(x)."""
    return mu + A * g(x, c, sigma)

def lnL(mu, A, D, x, c, sigma):
    """Log-verosimilitud para los datos D dados los parámetros mu y A."""
    F_vals = F(mu, A, x, c, sigma)
    # Evitar log(0) usando np.where para manejar casos donde F_vals es cero
    log_F = np.where(F_vals > 0, np.log(F_vals), -np.inf)
    return np.sum(D * log_F - F_vals)

def dlnL_dmu(mu, A, D, x, c, sigma):
    """Derivada de la log-verosimilitud respecto a mu."""
    F_vals = F(mu, A, x, c, sigma)
    return np.sum((D / F_vals) - 1.0)

def dlnL_dA(A, mu, D, x, c, sigma):
    """Derivada de la log-verosimilitud respecto a A."""
    g_vals = g(x, c, sigma)
    F_vals = F(mu, A, x, c, sigma)
    return np.sum(((D / F_vals) - 1.0) * g_vals)

def estimar_mle_individual(D, x, c, sigma, mu_0, A_0):
    """Realiza la estimación MLE para una sola muestra de datos D."""
    max_bound = max(10 * D.max() + 10, 100.0)
    
    # MLE de mu (fijando A = A_0)
    mu_verosimilitud = brentq(
        lambda mu: dlnL_dmu(mu, A_0, D, x, c, sigma), 
        1e-6, 
        max_bound
    )
    
    # MLE de A (fijando mu = mu_0)
    A_verosimilitud = brentq(
        lambda A: dlnL_dA(A, mu_0, D, x, c, sigma), 
        0.0, 
        max_bound
    )
    
    return mu_verosimilitud, A_verosimilitud

def mle(n_muestras, x, c, sigma, mu_rng, A_rng):
    mu_resultados = []
    A_resultados = []
    D_total = []
    
    for _ in range(n_muestras):
        # Generar datos de Poisson basados en los parámetros reales
        D = np.random.poisson(F(mu_rng, A_rng, x, c, sigma))
        D_total.append(D)
        
        # Estimar MLE usando los valores reales como referencia (mu_0, A_0)
        mu_mle, A_mle = estimar_mle_individual(D, x, c, sigma, mu_rng, A_rng)
        
        mu_resultados.append(mu_mle)
        A_resultados.append(A_mle)
        
    return np.array(mu_resultados), np.array(A_resultados), np.array(D_total)