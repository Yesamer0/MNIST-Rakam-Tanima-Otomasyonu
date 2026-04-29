import tensorflow as tf
import cv2
import numpy as np
import os

# 1. Modeli yüklüyoruz
model = tf.keras.models.load_model('el_yazisi_modeli.h5')

klasor_yolu = 'resimler'

print(f"'{klasor_yolu}' klasörü taranıyor (Otomatik Arka Plan Algılama Aktif)...\n")

for dosya_adi in os.listdir(klasor_yolu):
    if dosya_adi.endswith((".png", ".jpg", ".jpeg")):
        resim_tam_yol = os.path.join(klasor_yolu, dosya_adi)
        
        # Resmi oku (Gri tonlamalı)
        img = cv2.imread(resim_tam_yol, cv2.IMREAD_GRAYSCALE)
        
        # --- OTOMATİK ARKA PLAN KONTROLÜ ---
        # Resimdeki piksellerin ortalama parlaklığına bakıyoruz.
        # Eğer ortalama 127'den büyükse resim genel olarak beyazdır (beyaz kağıt).
        if np.mean(img) > 127:
            img = cv2.bitwise_not(img) # Beyaz kağıdı siyah yap (Zekanın sevdiği hal)
        
        # Standart işlemler
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28)
        
        # Tahmin
        tahmin = model.predict(img, verbose=0)
        sonuc = np.argmax(tahmin)
        
        print(f"Dosya: {dosya_adi:15} ==> Tahmin: {sonuc}")

print("\nİşlem tamamlandı!")