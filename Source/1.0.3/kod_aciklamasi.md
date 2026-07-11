# 1.0.2 sürümü kod açıklaması

Bu belge, 1.0.2 klasöründeki proje yapısını ve ana kod parçalarının ne işe yaradığını özetler.

## Proje amacı

Bu sürüm, basit bir kelime tabanlı dil modeli oluşturan küçük bir sinir ağı örneğidir. Amaç, verilen kelime dizilerinden sonraki kelimeyi tahmin etmektir. Daha sonra bu model, kullanıcıdan gelen girişlere benzer cevaplar üretmek için kullanılabilir.

## Ana dosyalar

### main.py
- Modeli oluşturur, eğitir ve çalıştırır.
- `createModel()` adlı fonksiyon ile bir `Network` nesnesi kurar.
- Eğitimi tamamladıktan sonra modeli `model.qai` dosyasına kaydeder.
- `startModel()` ise kullanıcıdan metin alır ve modelin tahminlerine göre cevap üretir.

### train_datas.py
- Kelime havuzunu (`words`) tanımlar.
- Her kelimeye bir sayı atar (`word2id`, `id2word`).
- Eğitilecek örnekleri `data` listesine ekler.
- `encode()` fonksiyonu bir metni sayılar dizisine dönüştürür.

### core/network.py
- Modelin ana çekirdeğini temsil eder.
- Aşağıdaki bileşenleri kullanır:
  - `Embedding`: kelimeleri vektörlere çevirir.
  - `SelfAttention`: kelime ilişkilerini yakalamaya çalışır.
  - Feed-forward katmanları: tahminleri işlemek için kullanılır.
- `forward()` ile bir giriş dizisinin çıktısını hesaplar.
- `train_step()` ile tek bir örnek üzerinde geri yayılım yapar.
- `train()` ile tüm veri üzerinde eğitim döngüsünü çalıştırır.
- `predict()` ile bir sonraki kelimeyi tahmin eder.

### core/embedding.py
- Kelimelerin vektör temsillerini tutar.
- Her kelime için rastgele başlangıç vektörü oluşturur.
- Eğitim sırasında bu vektörleri günceller.

### core/attention.py
- Basit bir self-attention mekanizması içerir.
- Her kelime için sorgu, anahtar ve değer vektörleri üretir.
- Bu sayede kelimeler arasındaki bağları modellemeye çalışır.

### core/layer.py
- Daha genel bir katman yapısı sağlar.
- Bu sürümde ana modelde doğrudan çok fazla kullanılmasa da, katman mantığını göstermektedir.

### utils/math_utils.py
- `softmax()` ve `cross_entropy()` gibi matematiksel yardımcı fonksiyonları içerir.
- Modelin olasılık hesaplamalarında kullanılır.

## Eğitim akışı

1. `main.py` çalıştırılır.
2. `Network` nesnesi oluşturulur.
3. `train_datas.py` içindeki örnekler modele verilir.
4. `Network.train()` tüm veriler üzerinde eğitim yapar.
5. Model eğitildikten sonra kaydedilir.
6. Kullanıcı girişi alındığında `predict()` ile sonraki kelime tahmin edilir.

## Model mantığı

Model, her kelimeyi önce bir vektöre dönüştürür. Daha sonra bu vektörlere konumsal bilgi ekler. Sonra attention ve feed-forward katmanlarından geçerek bir sonraki kelime için olasılık dağılımı üretir.

## Kısa özet

Bu proje, öğretici amaçlı yazılmış küçük bir dil modeli örneğidir. Özellikle:
- kelime temsilleri,
- attention mekanizması,
- eğitim ve tahmin akışı,
- basit veri işleme

konularını görsel olarak anlamak için uygundur.
