"""
100 Bagimsiz Test Goruntusu - Toplu Dogruluk Testi
====================================================
best.pt modelini TEST_100_BAGIMSIZ klasorundeki 100 gorsel uzerinde tek tek
calistirir, dosya adindan beklenen sinifi tahmin eder (dosya adlandirma
kuralimiz: siniftan_XXX.jpg), tespit sonucuyla karsilastirir ve ozet bir
rapor (CSV) uretir.

Kullanim:
    python toplu_test.py
    python toplu_test.py --model best.pt --test_dir TEST_100_BAGIMSIZ

Varsayilanlar: model="best.pt", test_dir="TEST_100_BAGIMSIZ" (script ile
ayni klasorde olmalilar). TEST_100_BAGIMSIZ_FINAL_paket.zip'i acip klasoru
bu isimle (ya da --test_dir ile kendi verdigin isimle) script'in yanina koy.

Uygulamadaki ile AYNI ayarlar kullanilir: conf=0.25, imgsz=1280.

Cikti: toplu_test_sonuclari.csv (her gorsel icin detay) ve konsola ozet tablo.
"""

import argparse
import csv
import os
import re
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("HATA: ultralytics kutuphanesi bulunamadi. Kurulum icin:")
    print("    pip install ultralytics")
    sys.exit(1)


GUVEN_ESIGI = 0.25
TAHMIN_GORUNTU_BOYUTU = 1280

SINIFLAR_UZUNDAN_KISAYA = ["hiz_siniri", "park_yasak", "yaya_gecidi", "yol_ver", "dur"]

GORUNTU_UZANTILARI = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def beklenen_sinifi_bul(dosya_adi):
    """Dosya adindan (kucuk harfe cevirip) beklenen sinifi cikarmaya calisir.
    Bulamazsa None doner (manuel kontrol gerektigi anlamina gelir)."""
    ad = dosya_adi.lower()
    for sinif in SINIFLAR_UZUNDAN_KISAYA:
        pattern = r'(?<![a-z0-9])' + re.escape(sinif) + r'(?![a-z0-9])'
        if re.search(pattern, ad):
            return sinif
    return None


def main():
    parser = argparse.ArgumentParser(description="100 bagimsiz test gorseli uzerinde toplu dogruluk testi")
    parser.add_argument("--model", default="best.pt", help="best.pt model dosyasinin yolu")
    parser.add_argument("--test_dir", default="TEST_100_BAGIMSIZ", help="Test gorsellerinin oldugu klasor")
    parser.add_argument("--cikti", default="toplu_test_sonuclari.csv", help="Sonuc CSV dosyasinin adi")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"HATA: Model dosyasi bulunamadi: {args.model}")
        sys.exit(1)
    if not os.path.isdir(args.test_dir):
        print(f"HATA: Test klasoru bulunamadi: {args.test_dir}")
        print("TEST_100_BAGIMSIZ_FINAL_paket.zip'i acip bu klasorun yanina koy,")
        print(f"ya da --test_dir ile dogru yolu goster.")
        sys.exit(1)

    print(f"Model yukleniyor: {args.model}")
    model = YOLO(args.model)

    dosyalar = sorted(
        f for f in os.listdir(args.test_dir)
        if f.lower().endswith(GORUNTU_UZANTILARI)
    )
    if not dosyalar:
        print(f"HATA: {args.test_dir} icinde gorsel bulunamadi.")
        sys.exit(1)

    print(f"{len(dosyalar)} gorsel bulundu. Tespit basliyor (conf={GUVEN_ESIGI}, imgsz={TAHMIN_GORUNTU_BOYUTU})...\n")

    satirlar = []
    # sinif -> {toplam, dogru, kacirildi, yanlis_sinif}
    ozet = {s: {"toplam": 0, "dogru": 0, "kacirildi": 0, "yanlis_sinif": 0} for s in SINIFLAR_UZUNDAN_KISAYA}
    manuel_kontrol_listesi = []

    for i, dosya_adi in enumerate(dosyalar, 1):
        yol = os.path.join(args.test_dir, dosya_adi)
        beklenen = beklenen_sinifi_bul(dosya_adi)

        sonuc = model.predict(source=yol, conf=GUVEN_ESIGI, imgsz=TAHMIN_GORUNTU_BOYUTU, verbose=False)[0]

        tespitler = []
        if sonuc.boxes is not None:
            for kutu in sonuc.boxes:
                sinif_id = int(kutu.cls[0])
                sinif_adi = model.names[sinif_id]
                guven = float(kutu.conf[0])
                tespitler.append((sinif_adi, guven))

        tespit_edilen_siniflar = {t[0] for t in tespitler}
        tespit_metni = "; ".join(f"{s}:{g:.2f}" for s, g in tespitler) if tespitler else "(tespit yok)"

        if beklenen is None:
            durum = "MANUEL KONTROL"
            manuel_kontrol_listesi.append(dosya_adi)
        else:
            ozet[beklenen]["toplam"] += 1
            if beklenen in tespit_edilen_siniflar:
                durum = "DOGRU"
                if len(tespit_edilen_siniflar) > 1:
                    durum = "DOGRU (fazladan tespit de var)"
                ozet[beklenen]["dogru"] += 1
            elif not tespitler:
                durum = "KACIRILDI (levha bulunamadi)"
                ozet[beklenen]["kacirildi"] += 1
            else:
                durum = "YANLIS SINIF"
                ozet[beklenen]["yanlis_sinif"] += 1

        satirlar.append({
            "dosya": dosya_adi,
            "beklenen_sinif": beklenen if beklenen else "?",
            "tespit_edilenler": tespit_metni,
            "durum": durum,
        })

        print(f"[{i}/{len(dosyalar)}] {dosya_adi} -> beklenen={beklenen or '?'} | {tespit_metni} | {durum}")

    # CSV'ye yaz (Excel'de Turkce karakterler duzgun gorunsun diye utf-8-sig)
    with open(args.cikti, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["dosya", "beklenen_sinif", "tespit_edilenler", "durum"])
        writer.writeheader()
        writer.writerows(satirlar)

    # Ozet tablo
    print("\n" + "=" * 70)
    print("OZET (sadece dosya adindan sinifi belli olan gorseller uzerinden)")
    print("=" * 70)
    print(f"{'Sinif':<14}{'Toplam':>8}{'Dogru':>8}{'Kacirildi':>11}{'Yanlis Sinif':>14}{'Dogruluk':>11}")

    genel_toplam = genel_dogru = 0
    for sinif in SINIFLAR_UZUNDAN_KISAYA:
        v = ozet[sinif]
        if v["toplam"] == 0:
            continue
        oran = v["dogru"] / v["toplam"] * 100
        print(f"{sinif:<14}{v['toplam']:>8}{v['dogru']:>8}{v['kacirildi']:>11}{v['yanlis_sinif']:>14}{oran:>10.1f}%")
        genel_toplam += v["toplam"]
        genel_dogru += v["dogru"]

    print("-" * 70)
    if genel_toplam:
        print(f"{'TOPLAM':<14}{genel_toplam:>8}{genel_dogru:>8}{'':>11}{'':>14}{genel_dogru/genel_toplam*100:>10.1f}%")

    if manuel_kontrol_listesi:
        print(f"\n{len(manuel_kontrol_listesi)} gorselin dosya adindan sinifi anlasilamadi (manuel kontrol gerekiyor):")
        for ad in manuel_kontrol_listesi:
            print(f"  - {ad}")

    print(f"\nDetayli sonuclar '{args.cikti}' dosyasina kaydedildi.")


if __name__ == "__main__":
    main()
