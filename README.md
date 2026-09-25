# YOLO ile Trafik Levhası Tespit Sistemi 🚦

Bu proje, görüntü işleme ve derin öğrenme (YOLOv8) teknolojileri kullanılarak Türkiye'deki standart trafik levhalarını gerçek zamanlı tespit eden bir masaüstü uygulamasıdır.

## Projenin Amacı
Otonom sürüş ve sürücü destek sistemlerinin (ADAS) temelini oluşturan trafik levhası algılama problemini çözmek üzere tasarlanmıştır. Ultralytics YOLOv8 mimarisi ile özel bir veri setinde eğitilmiş model, `customtkinter` ile geliştirilen modern arayüz sayesinde kullanıcı dostu bir deneyim sunmaktadır.

## Kullanılan Teknolojiler
- **Algoritma:** YOLO (You Only Look Once) 
- **Dil & Arayüz:** Python, CustomTkinter (UI)
- **Görüntü İşleme:** OpenCV, Pillow
- **Veri Manipülasyonu:** Pandas (Test Sonuçları)

## Veri Seti
Projede kullanılan eğitim veri setinin (337 görsel, 5 sınıf) orijinal boyutlarına ve etiketlemelerine **[Roboflow üzerinden erişebilirsiniz](https://app.roboflow.com/beyzanur-basaran/trafik_levhasi_tespit/3)**.
*(Depoda boyut sınırını aşmamak için 87MB'lık `EGITIM_SETI.zip` dosyası `.gitignore` ile dışarıda bırakılmıştır.)*

## Nasıl Çalıştırılır?

**1. Depoyu İndirin:**
```bash
git clone https://github.com/Beyza006/Trafik-Levhasi-Tanima-YOLO.git
cd Trafik-Levhasi-Tanima-YOLO
```

**2. Gerekli Kütüphaneleri Kurun:**
```bash
pip install -r requirements.txt
```

**3. Uygulamayı Başlatın:**
```bash
python masaustu_uygulama_final.py
```
> **Not:** `best.pt` ağırlık (weights) dosyası proje ile aynı klasörde olmalıdır. Eğer farklı bir klasördeyse, arayüz açıldıktan sonra "Model Seç" butonu ile modeli manuel olarak yükleyebilirsiniz.

## Proje Dosyaları
- `masaustu_uygulama_final.py`: Ana masaüstü arayüzü (GUI) ve tahmin kodları.
- `toplu_test.py`: Eğitilen modelin toplu test edilmesi için kullanılan script.
- `YOLO_Trafik_Levhasi_Egitim_v3.ipynb`: YOLO modelinin eğitim adımlarını içeren Jupyter Notebook.
- `best.pt`: Eğitim sonucunda elde edilen nihai ve en yüksek performanslı (best) model ağırlıkları.
- `toplu_test_sonuclari.csv`: Batch test metrikleri.
- `YOLO_arastirma_sunumu.pptx`: Araştırma, metodoloji ve proje sunumu.

---
*Bu proje, bilgisayarlı görü (computer vision) ve yapay zeka alanında geliştirilmiş portfolyo çalışmasıdır.*
