import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

# 1) SENTETIK VERI OLUSTURMA
np.random.seed(42)

true_mu = 150.0 
true_sigma = 10.0  
n_obs = 50 

data = true_mu + true_sigma * np.random.randn(n_obs)


plt.figure(figsize=(8, 5))
plt.hist(data, bins=12, edgecolor='black')
plt.axvline(true_mu, linestyle='--', label='Gerçek μ')
plt.xlabel("Gözlenen Parlaklık")
plt.ylabel("Frekans")
plt.title("Sentetik Gözlem Verisi")
plt.legend()
plt.show()



# 2) BAYESYEN FONKSIYONLAR
def log_likelihood(theta, data):
    mu, sigma = theta

    if sigma <= 0:
        return -np.inf

    return -0.5 * np.sum(((data - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))


def log_prior(theta):
    mu, sigma = theta

    # Ödevde verilen geniş prior sınırları
    if 0 < mu < 300 and 0 < sigma < 50:
        return 0.0

    return -np.inf


def log_probability(theta, data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, data)



# 3) MCMC CALISTIRMA
initial = np.array([140, 5])
n_walkers = 32
n_dim = 2
n_steps = 2000


pos = initial + 1e-4 * np.random.randn(n_walkers, n_dim)

sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, args=(data,))
sampler.run_mcmc(pos, n_steps, progress=True)


# 4) CHAIN GORSELLESTIRME
samples = sampler.get_chain()

fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)

labels = [r"$\mu$", r"$\sigma$"]

for i in range(n_dim):
    ax = axes[i]
    ax.plot(samples[:, :, i], alpha=0.5)
    ax.set_ylabel(labels[i])

axes[-1].set_xlabel("Adım")
plt.suptitle("MCMC Zincirleri")
plt.tight_layout()
plt.show()


# 5) BURN-IN SONRASI ORNEKLER
flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)

print("Düzleştirilmiş örnek sayısı:", len(flat_samples))


# 6) PARAMETRE OZET ISTATISTIKLERI
mu_samples = flat_samples[:, 0]
sigma_samples = flat_samples[:, 1]

mu_q16, mu_median, mu_q84 = np.percentile(mu_samples, [16, 50, 84])
sigma_q16, sigma_median, sigma_q84 = np.percentile(sigma_samples, [16, 50, 84])

mu_abs_error = abs(mu_median - true_mu)
sigma_abs_error = abs(sigma_median - true_sigma)

print("\n--- SONUCLAR ---")
print(f"Gerçek μ = {true_mu}")
print(f"Tahmin μ (median) = {mu_median:.4f}")
print(f"μ için %16 = {mu_q16:.4f}")
print(f"μ için %84 = {mu_q84:.4f}")
print(f"μ mutlak hata = {mu_abs_error:.4f}")

print()

print(f"Gerçek σ = {true_sigma}")
print(f"Tahmin σ (median) = {sigma_median:.4f}")
print(f"σ için %16 = {sigma_q16:.4f}")
print(f"σ için %84 = {sigma_q84:.4f}")
print(f"σ mutlak hata = {sigma_abs_error:.4f}")


fig = corner.corner(
    flat_samples,
    labels=[r"$\mu$ (Parlaklık)", r"$\sigma$ (Hata)"],
    truths=[true_mu, true_sigma],
    show_titles=True
)
plt.show()
