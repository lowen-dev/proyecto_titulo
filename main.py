import numpy as np
from mle import mle
from mco import mco
from plotting_utils import graficar_resultados, graficar_comparacion_mco, mostrar_graficos
from crlb import (evaluar_crlb, evaluar_mco, imprimir_reporte_1d,
                  imprimir_reporte_comparativo)

# --- Parámetros ---
c, sigma = 0.0, 1.62                 # c y sigma de ireland
mu_rng, A_rng = 114.0, 1434.0        # Valores reales
N_muestras = 100000                  # Repetir 1000 veces
x = np.linspace(-8.5, 8.5, 18)       # x de -8 a 9

# Obtener los 1000 resultados
mu_mle, A_mle, D_array = mle(N_muestras, x, c, sigma, mu_rng, A_rng)

reporte = evaluar_crlb(mu_mle, A_mle, mu_rng, A_rng, x, c, sigma)
imprimir_reporte_1d(reporte)

# --- MCO sobre los MISMOS datos D que el MLE ---
mu_mco, A_mco, _ = mco(x, c, sigma, mu_rng, A_rng, D_total=D_array)
reporte_mco = evaluar_mco(mu_mco, A_mco, mu_rng, A_rng, x, c, sigma)
imprimir_reporte_comparativo(reporte, reporte_mco)

# Mostrar estadísticas básicas de los resultados
print("\n--- Resultados de las simulaciones ---")
print(f"mu_verosimilitud -> Promedio: {mu_mle.mean():.4f} | Desv. Est: {mu_mle.std():.4f} | Sesgo promedio: {mu_mle.mean() - mu_rng:.4f}")
print(f"A_verosimilitud  -> Promedio: {A_mle.mean():.4f} | Desv. Est: {A_mle.std():.4f} | Sesgo promedio: {A_mle.mean() - A_rng:.4f}")
print(f"mu_mco           -> Promedio: {mu_mco.mean():.4f} | Desv. Est: {mu_mco.std():.4f} | Sesgo promedio: {mu_mco.mean() - mu_rng:.4f}")
print(f"A_mco            -> Promedio: {A_mco.mean():.4f} | Desv. Est: {A_mco.std():.4f} | Sesgo promedio: {A_mco.mean() - A_rng:.4f}")


D_verosimilitud = D_array.mean(axis=0)
mu_verosimilitud = mu_mle.mean()
A_verosimilitud = A_mle.mean()


graficar_resultados(
    x, D_verosimilitud, mu_verosimilitud, A_verosimilitud, mu_rng, A_rng, c, sigma,
    std_mu_mle=np.sqrt(reporte['var_mu_mle']),
    std_crlb_mu=np.sqrt(reporte['var_crlb_mu']),
    std_A_mle=np.sqrt(reporte['var_A_mle']),
    std_crlb_A=np.sqrt(reporte['var_crlb_A'])
)

graficar_comparacion_mco(
    x, D_verosimilitud, mu_mle, A_mle, mu_mco, A_mco, mu_rng, A_rng, c, sigma,
    std_crlb_mu=np.sqrt(reporte['var_crlb_mu']),
    std_crlb_A=np.sqrt(reporte['var_crlb_A']),
    std_teo_mco_mu=np.sqrt(reporte_mco['var_teo_mu']),
    std_teo_mco_A=np.sqrt(reporte_mco['var_teo_A'])
)

mostrar_graficos()   # único plt.show(): abre ambas figuras a la vez

