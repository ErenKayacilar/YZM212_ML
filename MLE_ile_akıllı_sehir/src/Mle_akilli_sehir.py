# =============================================================================
# BAĞIMLILIKLAR (Dependencies)
# =============================================================================
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import poisson

# =============================================================================
# BÖLÜM 2: Python ile Sayısal (Numerical) MLE
# =============================================================================

# Gözlemlenen Trafik Verisi (1 dakikada geçen araç sayısı)
# Bu veri seti, caddeden geçen araçların frekansını temsil eder.
traffic_data = np.array([12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15])


def negative_log_likelihood(lam, data):
    """
    Poisson dağılımı için Negatif Log-Likelihood hesaplar.

    Mantık: Olasılıkları çarpmak yerine logaritma alıp toplamak işlem kolaylığı sağlar.
    scipy.optimize 'minimize' fonksiyonunu kullandığı için fonksiyonu negatif ile çarptık.
    """
    n = len(data)
    # Log-likelihood formülü: l(λ) = -n*λ + sum(k_i)*ln(λ)
    # nll değişkeni, bu değerin negatifini tutar (minimize edilmek üzere).
    nll = -(-n * lam + np.sum(data) * np.log(lam))
    return nll


# Başlangıç tahmini (Initial Guess) parametre arama süreci için rastgele bir başlangıç noktasıdır.
initial_guess = 1.0

# minimize fonksiyonu, NLL değerini en küçük yapan lambda (lam) değerini bulur.
# bounds=[(0.001, None)] ile lambda'nın 0'dan büyük olması gerektiğini garanti ediyoruz.
result = opt.minimize(negative_log_likelihood, initial_guess, args=(traffic_data,), bounds=[(0.001, None)])

# Optimizasyon sonucunda elde edilen en iyi parametre (MLE)
lam_mle = result.x[0]
analitik_tahmin = np.mean(traffic_data)

print("--- Bölüm 2 Sonuçları ---")
print(f"Sayısal Tahmin (MLE lambda): {lam_mle:.4f}")
print(f"Analitik Tahmin (Ortalama): {analitik_tahmin:.4f}\n")

# =============================================================================
# BÖLÜM 3: Model Karşılaştırma ve Görselleştirme
# =============================================================================

# Grafik alanı oluşturuluyor
plt.figure(figsize=(10, 6))

# 1. Gerçek veri setinin histogramı
# density=True: Y eksenini olasılık (density) ölçeğine getirir, böylece PMF ile karşılaştırılabilir.
plt.hist(traffic_data, bins=np.arange(min(traffic_data), max(traffic_data) + 2) - 0.5,
         density=True, alpha=0.6, color='skyblue', label='Gerçek Veri Histogramı', edgecolor='black')

# 2. Bulunan lambda değeriyle Poisson Olasılık Kütle Fonksiyonu (PMF) çizimi
# x_vals: Veri aralığı boyunca araç sayılarını temsil eder.
x_vals = np.arange(min(traffic_data) - 2, max(traffic_data) + 3)
pmf_vals = poisson.pmf(x_vals, lam_mle)

# plt.step: Poisson kesikli bir dağılım olduğu için basamaklı (step) gösterim tercih edildi.
plt.step(x_vals, pmf_vals, where='mid', color='red', marker='o', linestyle='-',
         label=fr'Poisson PMF ($\hat{{\lambda}}$={lam_mle:.2f})')

# Görselleştirme Zorunlulukları (Eksen isimleri, Başlık, Gösterge)
plt.xlabel('Dakikadaki Araç Sayısı')  # plt.xlabel
plt.ylabel('Olasılık Yoğunluğu')  # plt.ylabel
plt.title('Bölüm 3: Trafik Verisi vs Poisson Modeli Uyum Analizi')  # plt.title
plt.legend()  # plt.legend
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# =============================================================================
# BÖLÜM 4: Gerçek Hayat Senaryosu - "Outlier" Analizi
# =============================================================================

# Veri setine "200" araçlık hatalı bir gözlem (outlier) eklenmesi
outlier_value = 200
traffic_data_outlier = np.append(traffic_data, outlier_value)

# Outlier sonrası yeni MLE değerinin sayısal olarak hesaplanması
result_outlier = opt.minimize(negative_log_likelihood, initial_guess, args=(traffic_data_outlier,),
                              bounds=[(0.001, None)])
lam_mle_outlier = result_outlier.x[0]

print("--- Bölüm 4: Outlier Analizi ---")
print(f"Orijinal MLE (Lambda): {lam_mle:.4f}")
print(f"Outlier Sonrası MLE (Lambda): {lam_mle_outlier:.4f}")
print(f"Değişim Oranı: %{((lam_mle_outlier - lam_mle) / lam_mle * 100):.2f}")

# KRİTİK NOT: MLE yöntemi tüm veriyi eşit ağırlıkla değerlendirdiği için outlier'a duyarlıdır.