1. Temel Tanımlar
Matris Manipülasyonu: Matrislerin oluşturulması, dönüştürülmesi ve çarpanlarına ayrıştırılması gibi işlemlerin bütünüdür. Makine öğrenmesinde veriler n×m boyutlu matrisler olarak temsil edilir; bu matrisler üzerinde yapılan cebirsel işlemler modelin eğitilmesini ve yorumlanmasını mümkün kılar.

Özdeğer (Eigenvalue): Kare bir A matrisi için, sıfırdan farklı bir v vektörü ve λ skaleri A·v = λ·v eşitliğini sağlıyorsa, λ bu matrisin bir özdeğeridir. Bir özvektör, A matrisiyle çarpıldığında yalnızca ölçeği değişen (λ katı kadar), yönü değişmeyen özel bir vektördür; λ ise bu ölçekleme miktarını veren özdeğerdir.

Özvektör (Eigenvector): Özvektörler yeni özellik uzayının yönlerini, özdeğerler ise bu yönlerdeki büyüklüğü (varyans miktarını) belirler.

2. Makine Öğrenmesi ile İlişkisi
Makine öğrenmesinde verinin içindeki gizli yapıyı açığa çıkarmak için matris ayrıştırma yöntemleri kullanılır. Bu noktada özdeğer ve özvektörler merkezi bir rol üstlenir.
Kovaryans Matrisi: Kovaryans matrisinin köşegen elemanları değişkenlerin varyansını, köşegen dışı elemanlar ise değişkenler arasındaki artıklığı (kovaryansı) gösterir. Boyut azaltmada amaç varyansı maksimize ederken kovaryansı minimize etmektir; bu da kovaryans matrisinin köşegenleştirilmesiyle, yani özdeğer ayrıştırmasıyla elde edilir.

PCA (Temel Bileşen Analizi): PCA, verinin kovaryans matrisinin özvektörlerini ve özdeğerlerini bulur. En büyük özdeğere karşılık gelen özvektör, verideki en yüksek varyans yönünü gösterir ve birinci temel bileşeni tanımlar; ardışık bileşenler birbirlerine dik biçimde azalan varyansları yakalar.

3. Kullanılan Yöntem ve Yaklaşımlar

PCA-EIG (Özdeğer Ayrıştırması ile PCA)

Standardize edilmiş veri setinin kovaryans matrisi hesaplanır; bu matris özvektörler ve özdeğerlere ayrıştırılarak verinin yayıldığı yönler ve büyüklükler elde edilir.Ancak bu yöntem simetrik ve kare matrislerde çalışır ve sayısal açıdan kararsız olabilir.

PCA-SVD (Tekil Değer Ayrıştırması ile PCA)

Herhangi bir m×n matrisi X = U · Σ · Vᵀ biçiminde üç matrisin çarpımına ayrıştırılır. SVD, özdeğer ayrıştırmasının aksine kare olmayan matrislere de uygulanabilir; ayrıca tekil değerler sayısal olarak özdeğerlerden daha kararlıdır.Bu nedenle PCA-SVD, günümüz makine öğrenmesi kütüphanelerinde çok daha yaygındır.

LDA (Doğrusal Ayırt Edici Analiz)

PCA tüm veri setini farklı bir özellik alt uzayına yansıtırken, LDA sınıflar arasındaki ayrımı maksimize eden yönleri bulmaya çalışır; bu da örüntü sınıflandırma problemlerinde kullanışlıdır

Öneri Sistemleri ve Görüntü İşleme
SVD; veri sıkıştırma, gürültü giderme, yüz tanıma ve öneri sistemlerinde işbirlikçi filtreleme gibi pek çok uygulamada kullanılmaktadır.

4. Kaynaklar
1. Hui, J. (2020). Machine Learning — Singular Value Decomposition (SVD) & Principal Component Analysis (PCA). Medium. https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491
2. ML and DL Explained. (2025). Singular Value Decomposition (SVD) vs. Eigen Decomposition: A Deep Dive. Medium. https://medium.com/@ml_dl_explained/singular-value-decomposition-svd-vs-eigen-decomposition-a-deep-dive-6cc99463b45d

5. numpy.linalg.eig Fonksiyonu: Dokümantasyon ve Kaynak Kod İncelemesi

2.1 Fonksiyon İmzası ve Parametreler
numpy.linalg.eig(a)
Parametre:
a → (..., M, M) boyutunda kare matris (veya matris yığını). Reel ya da karmaşık sayılardan oluşabilir.
Döndürdüğü Değerler (namedtuple olarak):
eigenvalues → (..., M) boyutunda dizi. Özdeğerler çokluklarına göre tekrarlanır ve sıralı olmak zorunda değildir. Sanal kısmı sıfır olan özdeğerler reel tipe dönüştürülür; gerçek matrisler için sonuçlar ya gerçek sayılardır ya da eşlenik çiftler halinde gelir

eigenvectors → (..., M, M) boyutunda dizi. eigenvectors[:,i] sütunu, eigenvalues[i] özdeğerine karşılık gelen normalize edilmiş (birim uzunluklu) özvektördür

Hata Durumu:
LinAlgError → Özdeğer hesabı yakınsamamazsa fırlatılır.

2.2 Matematiksel Tanım
w sayısı, a @ v = w * v eşitliğini sağlayan sıfırdan farklı bir v vektörü mevcutsa a'nın bir özdeğeridir. Başka bir deyişle tüm i için a @ eigenvectors[:,i] = eigenvalues[i] * eigenvectors[:,i] bağıntısı sağlanır

2.3 Kaynak Kod: Katmanlar ve LAPACK Bağlantısı
Kaynak kodun mimarisi üç katmandan oluşur:
Katman 1 — Python Arayüzü (_linalg.py)
Kaynak kodun Python tarafı numpy/linalg/_linalg.py dosyasında, #L1345–L1499 satırları arasında yer alır. 
Bu katmanda:
Girdi matrisi tip kontrolünden geçirilir ve gerekirse float64 ya da complex128'e çevrilir
Matrisin kare olup olmadığı doğrulanır
Matrisin gerçek mi karmaşık mı olduğu belirlenerek uygun LAPACK rutini seçilir
Çıktılar EigResult namedtuple'ı içine paketlenerek döndürülür

Katman 2 — C Uzantısı (umath_linalg.c)
Python katmanı doğrudan Fortran'ı çağırmaz; araya bir C uzantısı girer. Bu katman döngüsel işleri, bellek yönetimini ve veri tipi dağıtımını (dispatch) üstlenir.
Katman 3 — LAPACK Rutinleri (Fortran)
Fonksiyon, genel kare matrislerin özdeğer ve özvektörlerini hesaplayan _geev LAPACK rutinleri üzerine inşa edilmiştir.
Somut olarak:
Gerçek matrisler için → dgeev (double precision)
Karmaşık matrisler için → zgeev (complex double precision)
_geev rutinleri önce matrisi Hessenberg formuna indirgeyip ardından QR algoritmasıyla özdeğerleri bulur; bu yaklaşım karakteristik polinom yöntemine kıyasla sayısal açıdan çok daha kararlıdır.

2.4 Önemli Teknik Notlar
Sağ özvektörler döndürülür, sol değil.
y.T @ a = z * y.T eşitliğini sağlayan y vektörü sol özvektör olarak adlandırılır; genel olarak bir matrisin sol ve sağ özvektörleri birbirinin (eşlenik) transpozu değildir
Özvektör bağımsızlığı garanti değildir.
Özvektörler matrisi tam kerte olmayabilir; bazı sütunlar doğrusal bağımlı olabilir (yuvarlama hatası bunu gizleyebilir). Özdeğerlerin tamamı birbirinden farklıysa teorik olarak özvektörler doğrusal bağımsızdır ve inv(eigenvectors) @ a @ eigenvectors köşegendir.

Hermitian matrisler için eigh tercih edilmeli.
Hermitian olmayan normal matrislerde SciPy'ın scipy.linalg.schur fonksiyonu tercih edilir; çünkü bu fonksiyon özvektörler matrisinin uniter olduğunu garanti eder, ancak eig bunu garanti etmez

Yuvarlama hatalarına dikkat.
Belgelerde şu örnek verilmiştir: 1 ± 1e-9 teorik özdeğerli bir matris için eig, her iki özdeğeri de 1.0 olarak döndürebilir — bu, makine hassasiyetinin sınırıdır.

