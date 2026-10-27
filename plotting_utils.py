import numpy as np
import matplotlib.pyplot as plt
from mle import F, lnL

def graficar_resultados(x, D_ejemplo, mu_verosimilitud, A_verosimilitud, mu_rng, A_rng, c, sigma, 
                        std_mu_mle=None, std_crlb_mu=None, std_A_mle=None, std_crlb_A=None):
    """
    Genera la figura de 3 paneles con los datos, ajustes y comparación 
    gráfica de desviaciones estándar (MLE vs CRLB) en los paneles 2 y 3.
    """
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # 1) Datos y modelos
    xx = np.linspace(x.min(), x.max(), 400)

    ax[0].plot(x, D_ejemplo, "ko", label="Datos D")
    ax[0].plot(xx, F(mu_rng, A_rng, xx, c, sigma), "g-", label="Real")
    ax[0].plot(xx, F(mu_verosimilitud, A_verosimilitud, xx, c, sigma), "r--", label="MLE Fit")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("D")
    ax[0].legend()
    ax[0].set_title("Datos y ajustes")

    # 2) ln L vs mu (A = A_0)
    mu_grid = np.linspace(mu_verosimilitud - 10, mu_verosimilitud + 10, 300)
    lnL_mu_vals = [lnL(m, A_rng, D_ejemplo, x, c, sigma) for m in mu_grid]
    
    ax[1].plot(mu_grid, lnL_mu_vals, 'b-', lw=1.5, label="ln L(mu)")
    ax[1].axvline(mu_verosimilitud, color="r", ls="--", label=f"mu_mle = {mu_verosimilitud:.2f}")
    ax[1].axvline(mu_rng, color="g", ls=":", label=f"mu_real = {mu_rng}")

    # Mostrar comparación de desviaciones estándar en mu si se proporcionan
    if std_crlb_mu is not None and std_mu_mle is not None:
        # Sombreado o rangos de +- 1 sigma CRLB vs MLE alrededor del estimado
        ax[1].axvspan(mu_verosimilitud - std_crlb_mu, mu_verosimilitud + std_crlb_mu, 
                      color='orange', alpha=0.2, label=f"CRLB 1D (σ={std_crlb_mu:.2f})")
        ax[1].axvspan(mu_verosimilitud - std_mu_mle, mu_verosimilitud + std_mu_mle, 
                      color='red', alpha=0.1, label=f"MLE Emp. (σ={std_mu_mle:.2f})")

    ax[1].set_xlabel("mu")
    ax[1].set_ylabel("ln L")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"ln L(mu | A = {A_rng})")

    # 3) ln L vs A (mu = mu_0)
    A_grid = np.linspace(max(0.1, A_verosimilitud - 100), A_verosimilitud + 100, 300)
    lnL_A_vals = [lnL(mu_rng, a, D_ejemplo, x, c, sigma) for a in A_grid]

    ax[2].plot(A_grid, lnL_A_vals, 'b-', lw=1.5, label="ln L(A)")
    ax[2].axvline(A_verosimilitud, color="b", ls="--", label=f"A_mle = {A_verosimilitud:.2f}")
    ax[2].axvline(A_rng, color="g", ls=":", label=f"A_real = {A_rng}")

    # Mostrar comparación de desviaciones estándar en A si se proporcionan
    if std_crlb_A is not None and std_A_mle is not None:
        ax[2].axvspan(A_verosimilitud - std_crlb_A, A_verosimilitud + std_crlb_A, 
                      color='orange', alpha=0.2, label=f"CRLB 1D (σ={std_crlb_A:.2f})")
        ax[2].axvspan(A_verosimilitud - std_A_mle, A_verosimilitud + std_A_mle, 
                      color='blue', alpha=0.1, label=f"MLE Emp. (σ={std_A_mle:.2f})")

    ax[2].set_xlabel("A")
    ax[2].set_ylabel("ln L")
    ax[2].legend(fontsize=8)
    ax[2].set_title(f"ln L(A | mu = {mu_rng})")

    plt.tight_layout()
    plt.show()