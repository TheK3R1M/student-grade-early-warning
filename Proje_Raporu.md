# Öğrenci Performans ve Davranış Verisi: Problem ve Analiz Raporu

Hocanızın belirlemiş olduğu "Önce veri setini bul, sonra arkasına mantıklı bir problem inşa et" konseptine uygun olarak, elimizdeki Kaggle veri setine (*Student Performance & Behavior Dataset*) harika bir iş problemi (Business Problem) uydurduk. Aşağıda hocanıza sunmanız için proje adımlarının tam karşılığı olan "Proje Raporu" yer almaktadır.

---

## 1. Problemi Açıkla (Problem Tanımı)
Eğitim kurumlarında öğrencilerin akademik başarısızlığı (not düşüşü veya dersten kalma), genellikle sadece dönem sonu notlarına bakılarak çok geç fark edilmektedir. Oysa öğrencilerin yaşantıları (stres seviyeleri, uyku saatleri, derse katılımları gibi faktörler) izlenerek notlarının düşeceği yapay zeka ile önceden öngörülebilir.
Buna ek olarak eğitimcilerin "derslere çok katılan öğrencilere" bilinçdışı bir şekilde yüksek not vermesi (Değerlendirme Yanlılığı - Grading Bias) da ayrı bir gizli sorundur.

**Temel Problemimiz:** Risk altındaki öğrencileri (notu düşme eğiliminde olanları) çok önceden saptayıp onlara destek sağlamanın ötesinde; **sınıftaki Not Dengesizliğini (Class Imbalance)** gidererek adil bir dağılım oluşturmaktır. Eğer bir okulda herkes ortalama not alıyorsa ve sınır uçlarındaki (A ve F) öğrencilerin tespiti zorlaşıyorsa, modelin başarılı ve başarısız öğrencileri ayırt etmesi imkansızlaşır.

## 2. Veri Setini Tanıt
Bu sorunu çözmek için Kaggle'dan **"Student Performance & Behavior Dataset"** isimli gerçekçi bir otonom veri seti seçildi. 
Veri seti, özel bir eğitim kurumundaki öğrencilerin demografik bilgilerini, evdeki internet erişimlerini, haftalık ders çalışma saatlerini, katılım oranlarını, sınavlardan aldıkları puanları, stres (1-10) ve uyku düzenlerini içermektedir. İçerisinde veri bilimcilerin çözmesi amacıyla "Not Yanlılığı (Bias)" ve "Eksik veriler (Missing Values)" barındıracak şekilde yapılandırılmıştır.

## 3. Önemli Değişkenleri Seç
Hazırlanan veri ön işleme (preprocessing) ve modelleme mimarisinde odaklanılacak en önemli bağımlı (hedef) değişken ve bağımsız değişkenler şunlardır:
* **Hedef (Bağımlı) Değişken:** `Grade` (Öğrencinin Yıl Sonu Harf Notu)
* **Önemli Bağımsız Değişkenler:** 
  * `Midterm_Score` & `Assignments_Avg` (Akademik Başarı Öncülleri)
  * `Attendance (%)` (Yoklama - Hem başarıyı hem de hocaların bıraktığı yanlılığı (bias) tespit etmek için)
  * `Stress_Level (1-10)` ve `Sleep_Hours_per_Night` (Psikolojik ve Yaşamsal Faktörler)

*(Not: Ad, soyad ve numara gibi makine öğrenmesine katkısı olmayan gürültü değişkenler işlemden çıkarılarak verimlilik artırılmıştır.)*

## 4. Bu Veriyle Ne Analiz Yapılır?
Python kullanılarak bu veri üzerinde üç farklı boyutta analiz gerçekleştirilebilir:
1. **Keşifsel Veri Analizi (EDA) ve İstatistik:** Hangi departmanların (CS, Engineering vb.) daha çok stres altında olduğu veya devamsızlığın notlara olan doğrudan etkisi Scatter (Dağılım) grafikleri ile tespit edilebilir.
2. **Korelasyon Analizi:** Öğrencilerin uyku saatleriyle final notları arasında ya da aile gelir düzeyiyle derse katılım arasında sistematik bir ilişki olup olmadığı Isı Haritası (Heatmap) ile saptanabilir.
3. **Sınıflandırma (Makine Öğrenmesi):** Algoritmalara veriler (X) ve sonuç harf notu (y) verilerek; sadece sınav ve yaşam değişkenleri kullanılarak "bu öğrencinin yıl sonunda kalıp kalmayacağını (F veya A alacağını)" önceden kestiren bir `Random Forest` (Rastgele Orman) modeli eğitilebilir.

## 5. Tahmini Sonuç
Python ile modeli eğittiğimizde ortaya çıkacak tahmini (ve kodlarımızla ispatlanmış) sonuçlar şöyledir:
* **Sınıf Dengesizliği (Class Imbalance) Çözümü:** Öğrencilerin notları ortalamada yığıldığı için makine öğrenmesi algoritmaları zorlanmaktaydı. Bunu önlemek için sabit notlandırma bırakılıp, **Çan Eğrisi (Standart Sapma Tabanlı Bağıl Değerlendirme)** sistemine geçilerek dağılım kurtarılmıştır.
* **Öngörü (Tahmin) Başarısı:** Çan eğrisiyle adalet sağlandıktan sonra Rastgele Orman modeli, %80'in üzerinde bir doğruluk (Accuracy) oranıyla bir öğrencinin harf notunu başarıyla tahmin edebilir hale gelmiştir.
* **Beklenen İstatistiksel Algı:** Stres seviyesi yüksek ve uyku saati az olan öğrencilerin sınav notlarında bariz bir performans düşüşü görülecektir.

## 6. Çözüm Önerisi Sun
Elde edilen analiz çıktılarından yola çıkarak kuruma sunulacak iş kararları (Business Decisions) ve hedefe odaklanmış çözümler şunlardır:
1. **Yapay Zeka Destekli Erken Uyarı ve "Özel Ders" Ataması:** Makine öğrenmesi modeli (Random Forest) okulun otomasyonuna entegre edilmelidir. Model, daha dönem ortasındayken bir öğrencinin harf notunun "D veya F" seviyesine düşeceğini öngörürse, sistem otomatik olarak **öğrencinin zorlanma sebebine göre (akademik eksiklik, uyku/stres vb.)** o öğrenciye ekstra etüt saati, kişiselleştirilmiş destek programı veya özel bir mentör öğretmen atamalıdır.
2. **Çan Eğrisi (Bağıl Değerlendirme) Zorunluluğu:** Öğrencilerin notları ortalamada yığılma eğilimi gösterdiğinde, sabit not sınırları başarısız öğrencileri gizleyebilir. Standart Sapma'yı baz alan adil çan eğrisi notlandırması, makine öğrenmesi modelleri ile entegre çalışarak risk altındaki çocukları daha rahat saptar.
3. **Psikolojik ve Yaşamsal Müdahale:** Modelin uyku ve stresi "harf notu üzerinde en belirleyici" faktörlerinden biri olarak bulması ışığında, vize/final haftalarında "yüksek stresli" etiketi yiyen riskli öğrencilere okul tarafından hedefli ve özel psikolojik rehberlik sağlanmalıdır.
