# 📊 Öğrenci Performans Verisi: OSEMN Analitiği Raporu (Analitik Bakış Açısı)

Bu rapor, proje kapsamında gerçekleştirilen veri analitiği süreçlerini **OSEMN (Obtain, Scrub, Explore, Model, iNterpret)** çerçevesinde ele almaktadır. Bu çalışmanın temel amacı sadece teknik bir analiz yapmak değil, bir veri bilimci gözüyle **"Neyi, neden ve ne için yapıyoruz? Sistemde bir yanlılık (bias) var mı?"** sorularına yanıt aramaktır.

---

## 1. OBTAIN (Veriyi Elde Etme ve İlk Şüpheler)
Kaggle üzerinden sağlanan öğrenci performans veri seti ile çalışmaya başladık.
* **Başlangıç Veri Hacmi:** 5000 gözlem ve 23 değişken.

**Sorguladık:** Bu kadar fazla satıra sahip bir veri setinde her şeyin eksiksiz ve hatasız olması mümkün müdür? Eğer bir okuldaki tüm öğrencilerin verileri bir araya toplanmışsa, sistem kaynaklı veri kayıpları veya insan hataları beklemeli miyiz?

---

## 2. SCRUB (Veri Temizleme: Neden Silmedik de Doldurduk?)
Veri temizleme, "bozuk olanı at gitsin" mantığıyla yapılamaz. Verideki her boşluk, bize sistemin veya süreci yürütenlerin bir eksiği hakkında bilgi verir. 

### A. Eksik Verilerin Giderilmesi (Neden Silmedik?)
* **Yoklama (Attendance) & Ödevler (Assignments):** Yoklama sütununda 516, ödevlerde ise 517 adet veri eksikti. 
* **Analitik Karar:** Bu satırları tamamen silseydik, verimizin yaklaşık %10'unu kaybetmiş ve istatistiksel gücümüzü zayıflatmış olurduk. Belki de yoklamaya girmeyen öğrenciler sistemi kasıtlı olarak kullanmayanlardır? Bu yüzden verinin genel dağılımını (merkezi eğilimi) bozmamak adına bu boşlukları **Medyan (Ortanca)** yöntemiyle doldurduk.
* **Ebeveyn Eğitim Durumu:** 1794 adet boş veri mevcuttu. Bu sayı çok yüksek! Acaba kayıt formunda bu alan zorunlu tutulmamış mıydı? Sistem tasarımından kaynaklanan bu boşluğu, kategorik veri olduğu için **Mod (En Çok Tekrar Eden)** değer ile doldurarak yanlılığı en aza indirdik.

### B. Aykırı Değerler (Sistem Hatası mı, Gerçek Bir Süper Zeka mı?)
Not sistemlerinde (örneğin vize notu) çok uçuk değerler tespit ettik. Toplamda **0 adet** ekstrem değer IQR yöntemiyle sınırlandırıldı.
* **Neden Yaptık?:** Çünkü 100 üzerinden değerlendirilen bir sınavda matematiksel olarak imkansız değerler veya grubun genelinden inanılmaz derecede sapan notlar, makine öğrenmesi modelimizin algılamasını "yanıltır" (overfitting'e sürükler). Bu extremite, ya hocanın not okuma sistemindeki bir hatasıdır ya da optik okuyucu arızasıdır.

### C. Sınıf Dengesizliği (Class Imbalance) ve Çan Eğrisi Çözümü
Veri setinde öğrencilerin notları yoğunlukla ortalama etrafında toplandığı için, sabit (90=A, 80=B) not sistemi uygulandığında A ve F alan öğrenci sayısı modelin eğitemeyeceği kadar az oluyordu.
* **Şüphe:** "Sınıfta çok az başarılı ve çok az başarısız öğrenci var, model bunları göz ardı edip herkese orta halli diyor."
* **Analitik Karar:** Modelin başarılı ve risk altındaki öğrencileri daha net ayırt edebilmesi için sabit notlandırma sisteminden vazgeçip **Çan Eğrisi (Standart Sapma Tabanlı Bağıl Değerlendirme)** sistemine geçtik. Artık notlar, sınıfın genel başarı ve sapmasına göre istatistiksel olarak tamamen adil bir şekilde dağıtılıyor!

---

## 3. EXPLORE (Keşifsel Veri Analizi)
Sadece modeli eğitmeden önce, sistemin kendi içindeki "gizli hikayesini" görmek için görselleştirmeler (Plots) hazırladık:
* Demografik incelemelerde "Yaş vs Stres" eğilimlerine baktık. "Acaba yaş ilerledikçe sınav stresi kronik bir hal mi alıyor?" gibi sorulara yanıt aradık.

---

## 4. MODEL (Sadece Bir Karar Mekanizması Değil, Bir Açıklayıcı)
Veriyi düzelttikten sonra "Hangi öğrencinin hangi notu alacağını makineye öğretebilir miyiz?" diyerek **Random Forest** algoritmasını çalıştırdık.
* **Sonuç (Accuracy): %83.3**
  
### Peki Model Bize Neyi "Neden" Öneriyor? (Feature Importance)
Modelimizi salt bir "kara kutu (black box)" olarak sunmuyoruz. Aksine, modelin karar alırken nelere dikkat ettiğini inceliyoruz (Bkz: `06_osemn_feature_importance.png`). 
Modelin kararlarına göre, başarı üzerinde en etkili faktörlerin hangileri olduğunu görmek, bize şu soruyu sordurtuyor: *"Eğitim sistemimizde sadece final sınavlarına ağırlık vermek doğru mu? Öğrencilerin projelere veya stres yönetimine olan bağlılıkları başarılarını nasıl değiştiriyor?"*

---

## 5. INTERPRET (Bir Veri Bilimcisi Olarak Aksiyon Planı)
Bu proje kapsamında karşılaşılan veri sorunları ve uygulanan çözümler ışığında:
1. **Sistem Tasarımı Hatalıdır:** Orijinal veritabanındaki not hesaplama formülünün bozuk olduğu veya verilerin rastgele oluşturulduğu kesinleşmiştir. Sistem derhal düzeltilmelidir.
2. **Kayıt Formu Optimizasyonu:** Ebeveyn eğitim durumu gibi kritik bilgilerin bu kadar eksik kalmaması için okul kayıt sisteminde "Zorunlu Alan" doğrulamaları yapılmalıdır.
3. **Veriye Dayalı Karar:** %83.3 doğruluk payına sahip modelimiz, başarısız olma riski yüksek olan (özellikle stres seviyesi kritik olan) öğrencileri erkenden tespit etmek için aktif bir "Erken Uyarı Sistemi" olarak kullanılabilir.
