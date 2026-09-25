"""
YOLO ile Trafik Levhasi Tespit Uygulamasi
===========================================
Egitilmis best.pt modelini kullanarak yuklenen bir gorsel uzerinde
trafik levhalarini tespit eden basit bir masaustu uygulamasi.

Kullanilan kutuphaneler: customtkinter (arayuz), ultralytics (YOLO model),
Pillow (goruntu isleme).

Calistirmadan once:
    pip install customtkinter ultralytics pillow opencv-python

best.pt dosyasini bu script ile AYNI klasore koy (dosya adi tam olarak
"best.pt" olmali), ya da uygulama acilinca "Model Sec" ile elle goster.
"""

import os
import sys
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    messagebox.showerror(
        "Eksik Kutuphane",
        "ultralytics kutuphanesi bulunamadi.\nKurulum icin: pip install ultralytics"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
MODEL_DOSYA_ADI = "best.pt"
GUVEN_ESIGI = 0.25  # confidence threshold
TAHMIN_GORUNTU_BOYUTU = 1280  # tahmin sirasinda kullanilan cozunurluk (uzak/kucuk levhalari
                               # daha iyi yakalamak icin egitimdeki 640'tan daha yuksek tutuldu)
GORUNTU_ALANI_BOYUTU = (400, 400)  # goruntulerin ekranda gosterilecegi max boyut

# Sinif adlarini ekranda daha okunakli gostermek icin Turkce karsiliklari
SINIF_GORUNUM_ADLARI = {
    "dur": "Dur",
    "yol_ver": "Yol Ver",
    "hiz_siniri": "Hiz Siniri",
    "park_yasak": "Park Yasak",
    "yaya_gecidi": "Yaya Gecidi",
}


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class TrafikLevhasiUygulamasi(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("YOLO ile Trafik Levhasi Tespit Uygulamasi")
        self.geometry("1100x820")
        self.minsize(950, 760)

        self.model = None
        self.model_yolu = None
        self.secilen_gorsel_yolu = None

        self._arayuzu_olustur()
        self._modeli_yuklemeyi_dene()

    # ------------------------------------------------------------------
    # Arayuz kurulumu
    # ------------------------------------------------------------------
    def _arayuzu_olustur(self):
        ust_cerceve = ctk.CTkFrame(self, height=50, corner_radius=0)
        ust_cerceve.pack(side="top", fill="x")

        self.model_durum_etiketi = ctk.CTkLabel(
            ust_cerceve,
            text="Model yukleniyor...",
            font=ctk.CTkFont(size=13),
        )
        self.model_durum_etiketi.pack(side="left", padx=15, pady=10)

        model_sec_buton = ctk.CTkButton(
            ust_cerceve, text="Model Sec (best.pt)", width=160,
            command=self._model_sec
        )
        model_sec_buton.pack(side="right", padx=15, pady=10)

        govde = ctk.CTkFrame(self, fg_color="transparent")
        govde.pack(side="top", fill="both", expand=True, padx=15, pady=10)
        govde.grid_columnconfigure(0, weight=1)
        govde.grid_columnconfigure(1, weight=1)
        govde.grid_rowconfigure(0, weight=1)

        # --- Sol panel: secilen gorsel ---
        sol_cerceve = ctk.CTkFrame(govde)
        sol_cerceve.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        ctk.CTkLabel(
            sol_cerceve, text="Secilen Gorsel",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(pady=(12, 6))

        self.girdi_gorsel_etiketi = ctk.CTkLabel(
            sol_cerceve, text="Henuz gorsel secilmedi",
            width=GORUNTU_ALANI_BOYUTU[0], height=GORUNTU_ALANI_BOYUTU[1],
            fg_color=("gray85", "gray20"), corner_radius=8
        )
        self.girdi_gorsel_etiketi.pack(padx=15, pady=10)

        self.gorsel_sec_buton = ctk.CTkButton(
            sol_cerceve, text="Gorsel Sec / Yukle",
            command=self._gorsel_sec, height=38,
            font=ctk.CTkFont(size=14)
        )
        self.gorsel_sec_buton.pack(pady=(5, 8))

        self.tespit_buton = ctk.CTkButton(
            sol_cerceve, text="Nesne Tespitini Baslat",
            command=self._tespit_calistir, height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1f6f43", hover_color="#164f30",
            state="disabled"
        )
        self.tespit_buton.pack(pady=(0, 15))

        # --- Sag panel: sonuc gorseli + tespit listesi ---
        sag_cerceve = ctk.CTkFrame(govde)
        sag_cerceve.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        ctk.CTkLabel(
            sag_cerceve, text="Tespit Sonucu",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(pady=(12, 6))

        self.sonuc_gorsel_etiketi = ctk.CTkLabel(
            sag_cerceve, text="Sonuc burada gorunecek",
            width=GORUNTU_ALANI_BOYUTU[0], height=GORUNTU_ALANI_BOYUTU[1],
            fg_color=("gray85", "gray20"), corner_radius=8
        )
        self.sonuc_gorsel_etiketi.pack(padx=15, pady=10)

        ctk.CTkLabel(
            sag_cerceve, text="Tespit Edilen Trafik Levhalari:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=15, pady=(5, 2))

        self.sonuc_metin_kutusu = ctk.CTkTextbox(
            sag_cerceve, height=90, font=ctk.CTkFont(size=13)
        )
        self.sonuc_metin_kutusu.pack(fill="x", padx=15, pady=(0, 15))
        self.sonuc_metin_kutusu.insert("1.0", "-")
        self.sonuc_metin_kutusu.configure(state="disabled")

        self.alt_durum_etiketi = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=12), text_color="gray"
        )
        self.alt_durum_etiketi.pack(side="bottom", pady=(0, 8))

    # ------------------------------------------------------------------
    # Model yukleme
    # ------------------------------------------------------------------
    def _modeli_yuklemeyi_dene(self):
        varsayilan_yol = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), MODEL_DOSYA_ADI
        )
        if os.path.exists(varsayilan_yol):
            self._modeli_yukle(varsayilan_yol)
        else:
            self.model_durum_etiketi.configure(
                text="Model bulunamadi - sag ustten 'Model Sec' ile best.pt dosyasini goster",
                text_color="red"
            )

    def _model_sec(self):
        dosya_yolu = filedialog.askopenfilename(
            title="best.pt model dosyasini sec",
            filetypes=[("PyTorch model", "*.pt"), ("Tum dosyalar", "*.*")]
        )
        if dosya_yolu:
            self._modeli_yukle(dosya_yolu)

    def _modeli_yukle(self, dosya_yolu):
        try:
            self.model = YOLO(dosya_yolu)
            self.model_yolu = dosya_yolu
            self.model_durum_etiketi.configure(
                text=f"Model yuklendi: {os.path.basename(dosya_yolu)}",
                text_color=("black", "white")
            )
            if self.secilen_gorsel_yolu:
                self.tespit_buton.configure(state="normal")
        except Exception as e:
            self.model = None
            self.model_durum_etiketi.configure(
                text="Model yuklenemedi!", text_color="red"
            )
            messagebox.showerror("Model Hatasi", f"Model yuklenirken hata olustu:\n{e}")

    # ------------------------------------------------------------------
    # Gorsel secme
    # ------------------------------------------------------------------
    def _gorsel_sec(self):
        dosya_yolu = filedialog.askopenfilename(
            title="Bir gorsel sec",
            filetypes=[
                ("Gorsel dosyalari", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Tum dosyalar", "*.*"),
            ]
        )
        if not dosya_yolu:
            return

        self.secilen_gorsel_yolu = dosya_yolu
        self._gorseli_etikette_goster(dosya_yolu, self.girdi_gorsel_etiketi)

        self.sonuc_gorsel_etiketi.configure(image=None, text="Sonuc burada gorunecek")
        self._sonuc_metnini_guncelle("-")
        self.alt_durum_etiketi.configure(text="")

        if self.model is not None:
            self.tespit_buton.configure(state="normal")
        else:
            self.alt_durum_etiketi.configure(
                text="Once bir model yuklemelisin (sag ust - Model Sec).",
                text_color="orange"
            )

    def _gorseli_etikette_goster(self, dosya_yolu_veya_pil, etiket_widget):
        if isinstance(dosya_yolu_veya_pil, str):
            pil_gorsel = Image.open(dosya_yolu_veya_pil).convert("RGB")
        else:
            pil_gorsel = dosya_yolu_veya_pil

        pil_gorsel_kopya = pil_gorsel.copy()
        pil_gorsel_kopya.thumbnail(GORUNTU_ALANI_BOYUTU)

        ctk_gorsel = ctk.CTkImage(
            light_image=pil_gorsel_kopya,
            dark_image=pil_gorsel_kopya,
            size=pil_gorsel_kopya.size
        )
        etiket_widget.configure(image=ctk_gorsel, text="")
        etiket_widget.image = ctk_gorsel

    # ------------------------------------------------------------------
    # Tespit calistirma
    # ------------------------------------------------------------------
    def _tespit_calistir(self):
        if self.model is None:
            messagebox.showwarning("Model Yok", "Once bir model (best.pt) yuklemelisin.")
            return
        if not self.secilen_gorsel_yolu:
            messagebox.showwarning("Gorsel Yok", "Once bir gorsel secmelisin.")
            return

        self.alt_durum_etiketi.configure(text="Tespit yapiliyor, lutfen bekle...", text_color="gray")
        self.update_idletasks()

        try:
            sonuclar = self.model.predict(
                source=self.secilen_gorsel_yolu,
                conf=GUVEN_ESIGI,
                imgsz=TAHMIN_GORUNTU_BOYUTU,
                verbose=False,
            )
        except Exception as e:
            messagebox.showerror("Tespit Hatasi", f"Tespit sirasinda hata olustu:\n{e}")
            self.alt_durum_etiketi.configure(text="")
            return

        sonuc = sonuclar[0]

        cizilmis_bgr = sonuc.plot()
        cizilmis_rgb = cv2.cvtColor(cizilmis_bgr, cv2.COLOR_BGR2RGB)
        cizilmis_pil = Image.fromarray(cizilmis_rgb)

        self._gorseli_etikette_goster(cizilmis_pil, self.sonuc_gorsel_etiketi)

        tespit_edilenler = []
        if sonuc.boxes is not None and len(sonuc.boxes) > 0:
            for kutu in sonuc.boxes:
                sinif_id = int(kutu.cls[0])
                sinif_adi = self.model.names[sinif_id]
                guven = float(kutu.conf[0])
                gorunen_ad = SINIF_GORUNUM_ADLARI.get(sinif_adi, sinif_adi)
                tespit_edilenler.append((gorunen_ad, guven))

        if tespit_edilenler:
            metin = "Tespit edilen trafik levhalari:\n"
            for ad, guven in tespit_edilenler:
                metin += f"- {ad}  (guven: %{guven * 100:.0f})\n"
        else:
            metin = "Trafik levhasi tespit edilemedi."

        self._sonuc_metnini_guncelle(metin)
        self.alt_durum_etiketi.configure(
            text=f"Tamamlandi - {len(tespit_edilenler)} levha bulundu.",
            text_color=("black", "white")
        )

    def _sonuc_metnini_guncelle(self, metin):
        self.sonuc_metin_kutusu.configure(state="normal")
        self.sonuc_metin_kutusu.delete("1.0", "end")
        self.sonuc_metin_kutusu.insert("1.0", metin)
        self.sonuc_metin_kutusu.configure(state="disabled")


if __name__ == "__main__":
    uygulama = TrafikLevhasiUygulamasi()
    uygulama.mainloop()
