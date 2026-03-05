# Yapan Öğrenci:
Süleyman Eren Kayacılar 
23291547

# HMM ile İzole Kelime Tanıma

## Problem Tanımı
Bu projenin amacı, kısıtlı bir kelime dağarcığı (EV ve OKUL) üzerinden ses sinyallerini analiz ederek doğru kelimeyi tahmin edebilen bir sistem simüle etmektir. Temel problem, zamana bağlı ve olasılıksal bir yapı sunan ses verisinden (fonemler), gözlemlenen frekans karakteristikleri aracılığıyla en olası kelimeyi ve fonem dizisini deşifre etmektir.

## Veri
Proje kapsamında kullanılan veriler, ses spektrumunu temsil eden sembolik gözlemlerden oluşmaktadır:
- **Gizli Durumlar (States):** {e, v} fonemleri.
- **Gözlemler (Observations):** {High, Low} frekans karakteristikleri.
- **Parametreler:** Ödev kapsamında verilen Başlangıç (Initial), Geçiş (Transition) ve Emisyon (Emission) olasılık matrisleri eğitim verisi olarak kullanılmıştır.

## Yöntem
Sistem iki ana aşamadan oluşmaktadır:
1. **Sınıflandırma (Classification):** Her kelime için ayrı bir Kategorik Gizli Markov Modeli (Categorical HMM) tanımlanmıştır. Test verisi geldiğinde `score` fonksiyonu ile her modelin Log-Likelihood değeri hesaplanmakta ve en yüksek olasılığa sahip model "tahmin edilen kelime" olarak seçilmektedir.
2. **Deşifre Etme (Decoding):** Viterbi Algoritması kullanılarak, verilen gözlem dizisi için en olası gizli durum (fonem) dizisi elde edilmektedir.

## Sonuçlar
Girdi olarak verilen [High, Low] gözlem dizisi üzerinde yapılan testlerde:
- **EV Modeli Log-Likelihood:** -0.9729
- **OKUL Modeli Log-Likelihood:** -1.3863
- **Karar:** EV modeli daha yüksek olasılık verdiği için ses verisi "EV" olarak sınıflandırılmıştır.
- **Viterbi Sonucu:** En olası fonem dizisi ['e', 'v'] olarak bulunmuştur.

## Yorum ve Tartışma
Elde edilen sonuçlar, teorik el hesaplamalarıyla tam uyum göstermektedir. HMM yapısı, ses verisindeki zaman serisi belirsizliğini yönetmekte başarılıdır; ancak gözlem dizisindeki gürültü miktarının artması emisyon olasılıklarını etkileyerek modeller arasındaki farkı azaltabilir. Binlerce kelimelik geniş sistemlerde, HMM'in Markov kısıtlamaları ve manuel özellik tanımı gereksinimi nedeniyle Derin Öğrenme mimarileri (RNN, LSTM, Transformer) daha ölçeklenebilir bir alternatif sunmaktadır.

## Kurulum ve Çalıştırma

Gerekli kütüphaneleri kurmak için terminal üzerinden şu komutu kullanın:
```bash
pip install -r requirements.txt

Sistemi test etmek için ana dizinde şu komutu çalıştırın:
python src/First_assignment.py