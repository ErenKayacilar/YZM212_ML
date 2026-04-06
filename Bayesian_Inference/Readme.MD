Proje Özeti

Bu projede, sentetik gözlem verileri kullanılarak bir gök cisminin gerçek parlaklığı (μ) ve ölçüm hatası (σ) Bayesyen çıkarım yöntemleri ile tahmin edilmiştir. Amaç, gürültü içeren verilerden yola çıkarak bu iki bilinmeyen parametreyi olasılıksal olarak modellemek ve en olası değerlerini belirlemektir.

Projede, verilerin normal dağılımdan geldiği varsayılmış ve bu doğrultuda bir olasılık modeli kurulmuştur. Bayes Teoremi kullanılarak parametrelerin posterior dağılımı elde edilmiş, bu dağılımdan örnekleme yapmak için Markov Chain Monte Carlo (MCMC) yöntemi uygulanmıştır. MCMC örnekleme işlemi için emcee kütüphanesi kullanılmıştır.

Kodun ilk aşamasında, belirli bir ortalama (μ = 150) ve standart sapma (σ = 10) değerlerine sahip sentetik veriler oluşturulmuştur. Daha sonra, bu veriler için log-likelihood, log-prior ve log-posterior fonksiyonları tanımlanmıştır. Bu fonksiyonlar, modelin veriye ne kadar uyduğunu ve parametrelerin hangi aralıklarda olabileceğini belirlemek için kullanılmıştır.

emcee.EnsembleSampler kullanılarak çok sayıda örnekleme yapılmış ve elde edilen zincirlerden (chain) burn-in sonrası değerler alınarak analiz gerçekleştirilmiştir. Parametreler için medyan (median), alt (%16) ve üst (%84) güven aralıkları hesaplanmıştır.

Sonuçların görselleştirilmesi için matplotlib ve corner kütüphaneleri kullanılmıştır. Histogram grafikleri ile veri dağılımı incelenmiş, zincir grafikleri ile MCMC sürecinin kararlılığı gözlemlenmiş ve corner plot ile parametrelerin dağılımları ve aralarındaki ilişkiler analiz edilmiştir.

Elde edilen sonuçlar, modelin gürültülü veriye rağmen gerçek parametrelere oldukça yakın tahminler üretebildiğini göstermektedir. Bu çalışma, Bayesyen çıkarımın belirsizlik içeren veriler üzerinde güçlü ve esnek bir yöntem olduğunu ortaya koymaktadır.