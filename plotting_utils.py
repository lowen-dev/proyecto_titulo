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

# ==========================================================
# COMPARACIÓN MLE vs MCO (3 paneles)
# ==========================================================

def _gauss_pdf(z, media, std):
    return np.exp(-(z - media)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))


def graficar_comparacion_mco(x, D_ejemplo, mu_mle, A_mle, mu_mco, A_mco,
                             mu_rng, A_rng, c, sigma,
                             std_crlb_mu, std_crlb_A,
                             std_teo_mco_mu=None, std_teo_mco_A=None):
    """
    Figura de 3 paneles para comparar MLE vs MCO frente a la CRLB:
      1) Datos promedio y ajustes (real, MLE, MCO).
      2) Distribución de las estimaciones de mu (MLE vs MCO) y campana de la CRLB.
      3) Distribución de las estimaciones de A  (MLE vs MCO) y campana de la CRLB.
    mu_mle, A_mle, mu_mco, A_mco son los arrays de todas las simulaciones.
    """
    std_mu_mle, std_A_mle = np.std(mu_mle, ddof=1), np.std(A_mle, ddof=1)
    std_mu_mco, std_A_mco = np.std(mu_mco, ddof=1), np.std(A_mco, ddof=1)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # 1) Datos y ajustes
    xx = np.linspace(x.min(), x.max(), 400)
    ax[0].plot(x, D_ejemplo, "ko", label="Datos D (promedio)")
    ax[0].plot(xx, F(mu_rng, A_rng, xx, c, sigma), "g-", label="Real")
    ax[0].plot(xx, F(mu_mle.mean(), A_mle.mean(), xx, c, sigma), "r--", label="MLE Fit")
    ax[0].plot(xx, F(mu_mco.mean(), A_mco.mean(), xx, c, sigma), "b:", lw=2, label="MCO Fit")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("D")
    ax[0].legend()
    ax[0].set_title("Datos y ajustes: MLE vs MCO")

    # 2) y 3) Distribuciones de los estimadores
    paneles = [
        (ax[1], "mu", mu_mle, mu_mco, mu_rng, std_crlb_mu, std_mu_mle, std_mu_mco, std_teo_mco_mu),
        (ax[2], "A",  A_mle,  A_mco,  A_rng,  std_crlb_A,  std_A_mle,  std_A_mco,  std_teo_mco_A),
    ]
    for a, nombre, est_mle, est_mco, real, s_crlb, s_mle, s_mco, s_teo in paneles:
        lim = 4.5 * max(s_mle, s_mco)
        bins = np.linspace(real - lim, real + lim, 60)
        a.hist(est_mle, bins=bins, density=True, color="red", alpha=0.35,
               label=f"MLE (σ={s_mle:.2f})")
        a.hist(est_mco, bins=bins, density=True, color="blue", alpha=0.35,
               label=f"MCO (σ={s_mco:.2f})")

        zz = np.linspace(real - lim, real + lim, 400)
        a.plot(zz, _gauss_pdf(zz, real, s_crlb), color="orange", lw=2,
               label=f"CRLB 1D (σ={s_crlb:.2f})")
        if s_teo is not None:
            a.plot(zz, _gauss_pdf(zz, real, s_teo), color="navy", ls="--", lw=1.2,
                   label=f"MCO teórico (σ={s_teo:.2f})")

        a.axvline(real, color="g", ls=":", label=f"{nombre}_real = {real}")
        a.set_xlabel(nombre)
        a.set_ylabel("Densidad")
        a.legend(fontsize=8)
        a.set_title(f"Distribución de {nombre}: σ_MLE/σ_MCO = {s_mle/s_mco:.3f}")

    plt.tight_layout()
    plt.show()