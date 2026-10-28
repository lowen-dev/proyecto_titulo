import numpy as np
from mle import mle
from plotting_utils import graficar_resultados
from crlb import evaluar_crlb, imprimir_reporte_1d

def main():
    # --- Parámetros ---
    c, sigma = 0.0, 1.62                 # c y sigma de ireland
    mu_rng, A_rng = 114.0, 1434.0        # Valores reales
    N_muestras = 100                  # Repetir 1000 veces
    x = np.linspace(-8.5, 8.5, 18)       # x de -8 a 9
    
    # Obtener los 1000 resultados
    mu_array, A_array, D_array = mle(N_muestras, x, c, sigma, mu_rng, A_rng)

    reporte = evaluar_crlb(mu_array, A_array, mu_rng, A_rng, x, c, sigma)
    imprimir_reporte_1d(reporte)

    # Mostrar estadísticas básicas de los 1000 resultados
    print("\n--- Resultados de las 1000 simulaciones ---")
    print(f"mu_verosimilitud -> Promedio: {mu_array.mean():.4f} | Desv. Est: {mu_array.std():.4f}")
    print(f"A_verosimilitud  -> Promedio: {A_array.mean():.4f} | Desv. Est: {A_array.std():.4f}")


    D_verosimilitud = D_array.mean(axis=0)
    mu_verosimilitud = mu_array.mean()
    A_verosimilitud = A_array.mean()


    graficar_resultados(
        x, D_verosimilitud, mu_verosimilitud, A_verosimilitud, mu_rng, A_rng, c, sigma,
        std_mu_mle=np.sqrt(reporte['var_mu_mle']),
        std_crlb_mu=np.sqrt(reporte['var_crlb_mu']),
        std_A_mle=np.sqrt(reporte['var_A_mle']),
        std_crlb_A=np.sqrt(reporte['var_crlb_A'])
    )


main()