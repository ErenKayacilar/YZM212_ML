# MLE ile Akıllı Şehir Trafik Planlaması

Bu proje, bir belediyenin ulaşım departmanı için yoğun bir ana caddedeki trafik yoğunluğunu **Maximum Likelihood Estimation (MLE)** yöntemi kullanarak modellemek amacıyla geliştirilmiştir. Veri seti, birer dakikalık aralıklarla geçen araç sayılarını içermektedir.

## 1. Problem Tanımı
Şehir planlamasında yol genişletme veya sinyalizasyon kararlarını doğru verebilmek için trafik yoğunluğunun matematiksel bir modeline ihtiyaç duyulmaktadır. Bu projede, araç geçiş sayılarının **Poisson Dağılımı**'na uygun olduğu varsayılarak en uygun yoğunluk parametresi (lambda) tahmin edilmiştir.

## 2. Veri Seti
Analiz için kullanılan gözlemlenen trafik verisi (dakika başına araç sayısı):
`[12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15]`

## 3. Yöntem
Proje iki aşamalı bir çözüm sunar:
* **Teorik Çözüm:** Poisson dağılımı için Olabilirlik (Likelihood) fonksiyonu türetilmiş ve $\hat{\lambda}_{MLE}$ değerinin verilerin aritmetik ortalamasına eşit olduğu kanıtlanmıştır.
* **Sayısal Çözüm:** Python'da `scipy.optimize` kütüphanesi kullanılarak Negatif Log-Likelihood (NLL) fonksiyonu minimize edilmiş ve sayısal parametre tahmini yapılmıştır.

## 4. Sonuçlar
Yapılan analizler sonucunda şu değerler elde edilmiştir:
* Sayısal Tahmin (MLE lambda): 12.1429
* Analitik Tahmin (Ortalama): 12.1429
* Model Uyumu: Görselleştirme sonucunda Poisson modelinin gerçek verilerle yüksek uyum sağladığı gözlemlenmiştir.

## 5. Tartışma ve Aykırı Değer (Outlier) Analizi
Veri setine hatalı bir gözlem olan "200" değeri eklendiğinde şunlar gözlemlenmiştir:
* lambda değeri **12.14**'ten **24.66**'ya yükselmiş (%103 sapma).
* Tartışma: MLE yöntemi uç değerlere (outliers) karşı son derece hassastır. Tek bir hatalı veri, trafik yoğunluğunu olduğundan çok daha yüksek göstererek belediyenin gereksiz altyapı yatırımları yapmasına ve kaynak israfına yol açabilir. Bu durum, veri temizliğinin ve robust istatistiksel yöntemlerin önemini ortaya koymaktadır.

## Kullanılan Kütüphaneler
* `numpy`
* `scipy`
* `matplotlib`