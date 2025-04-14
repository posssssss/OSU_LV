import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

# Nakon transformacije slike u 2D polje
w, h, d = img.shape
img_array = np.reshape(img, (w*h, d))

# Prebrojavanje jedinstvenih boja
unique_colors = np.unique(img_array, axis=0)
num_unique_colors = len(unique_colors)
print(f"Broj jedinstvenih boja u slici: {num_unique_colors}")

# Primjena K-means algoritma za kvantizaciju na 5 boja
n_colors = 5
kmeans = KMeans(n_clusters=n_colors, random_state=0)
kmeans.fit(img_array)

# Zamjena originalnih boja s najbližim centroidima
labels = kmeans.predict(img_array)
centers = kmeans.cluster_centers_
img_array_aprox = centers[labels]

# Preoblikovanje natrag u originalnu dimenziju slike
img_quantized = img_array_aprox.reshape(w, h, d)

# Prikaz originalne i kvantizirane slike
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Originalna slika")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Kvantizirana slika ({n_colors} boja)")
plt.imshow(np.clip(img_quantized, 0, 1))
plt.axis('off')
plt.tight_layout()
plt.show()

#2)
# Određivanje broja jedinstvenih boja u originalnoj slici
unique_colors = np.unique(img_array, axis=0)
print(f"Broj jedinstvenih boja u originalnoj slici: {len(unique_colors)}")

# Primjena K-means algoritma za grupiranje boja
k = 5  # Broj željenih boja nakon kvantizacije
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(img_array)

# Zamjena originalnih RGB vrijednosti s najbližim centroidima klastera
labels = kmeans.predict(img_array)
img_array_aprox = kmeans.cluster_centers_[labels]

# Transformacija natrag u originalni oblik slike
img_aprox = img_array_aprox.reshape(w, h, d)

# Prikaz originalne i kvantizirane slike
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Originalna slika")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Kvantizirana slika ({k} boja)")
plt.imshow(np.clip(img_aprox, 0, 1))  # Osiguravanje da su vrijednosti između 0 i 1
plt.axis('off')
plt.tight_layout()
plt.show()

# Prikaz pronađene palete boja
plt.figure(figsize=(8, 2))
plt.title(f"Paleta od {k} boja")
for i, color in enumerate(kmeans.cluster_centers_):
    plt.fill([i, i+1, i+1, i], [0, 0, 1, 1], color=color)
plt.xlim(0, k)
plt.yticks([])
plt.tight_layout()
plt.show()

#3)
# Nakon primjene K-means algoritma (korak 2)
labels = kmeans.predict(img_array)

# Zamjena originalnih vrijednosti s centroidima klastera
img_array_aprox = kmeans.cluster_centers_[labels]

# Rekonstrukcija slike u originalne dimenzije
img_aprox = img_array_aprox.reshape(img.shape)

# Prikaz kvantizirane slike
plt.figure()
plt.title(f"Kvantizirana slika ({k} boja)")
plt.imshow(np.clip(img_aprox, 0, 1))  # Osiguravanje ispravnog raspona boja
plt.axis('off')
plt.show()

# Primjer kompletnog koda za sva tri zadatka
k = 5  # Broj željenih boja
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(img_array)

# Zamjena vrijednosti (zadatak 3)
labels = kmeans.predict(img_array)
img_array_aprox = kmeans.cluster_centers_[labels]
img_aprox = img_array_aprox.reshape(img.shape)

# Usporedba originala i kvantizirane slike
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Originalna slika")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Kvantizirana slika ({k} boja)")
plt.imshow(np.clip(img_aprox, 0, 1))
plt.axis('off')
plt.show()

#4)
# Generiranje usporedbe za različite vrijednosti K
k_values = [2, 5, 10, 16]
plt.figure(figsize=(15, 10))

# Originalna slika
plt.subplot(1, len(k_values)+1, 1)
plt.title("Originalna slika")
plt.imshow(img)
plt.axis('off')

# Kvantizirane slike
for i, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(img_array)
    img_quantized = kmeans.cluster_centers_[labels].reshape(img.shape)
    
    plt.subplot(1, len(k_values)+1, i+2)
    plt.title(f"K={k}")
    plt.imshow(np.clip(img_quantized, 0, 1))
    plt.axis('off')

plt.tight_layout()
plt.show()

#5)
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.image import imread

# Direktorij koji sadrži slike
directory = "imgs"

# Funkcija za kvantizaciju boje na svim slikama u direktoriju
def process_images(directory, k):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            filepath = os.path.join(directory, filename)
            
            # Učitavanje slike
            img = imread(filepath)
            
            # Normalizacija vrijednosti piksela (raspon 0-1)
            img = img.astype(np.float64) / 255
            
            # Pretvorba slike u 2D polje (jedan red su RGB komponente piksela)
            w, h, d = img.shape
            img_array = np.reshape(img, (w * h, d))
            
            # Primjena K-means algoritma
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(img_array)
            img_array_aprox = kmeans.cluster_centers_[labels]
            
            # Rekonstrukcija slike u originalne dimenzije
            img_quantized = img_array_aprox.reshape(w, h, d)
            
            # Prikaz originalne i kvantizirane slike
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title(f"Originalna slika: {filename}")
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title(f"Kvantizirana slika ({k} boja): {filename}")
            plt.imshow(np.clip(img_quantized, 0, 1))
            plt.axis('off')

            plt.tight_layout()
            plt.show()

# Primjena kvantizacije na sve slike u direktoriju s K=5 boja
process_images(directory, k=5)

#6)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib.image import imread

# Učitavanje slike i priprema podataka
img = imread("imgs/test_1.jpg")
img = img.astype(np.float64) / 255  # Normalizacija piksela
w, h, d = img.shape
img_array = np.reshape(img, (w * h, d))  # Transformacija u 2D polje

# Metoda lakta: izračunavanje inercije za različite vrijednosti K
inertia = []
k_values = range(1, 11)  # Testiranje broja grupa od 1 do 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img_array)
    inertia.append(kmeans.inertia_)  # Sprema inerciju za svaki broj klastera

# Grafički prikaz ovisnosti inercije o broju grupa K
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel("Broj grupa (K)")
plt.ylabel("Inercija (J)")
plt.title("Metoda lakta: Ovisnost inercije o broju grupa")
plt.grid(True)
plt.show()

#7)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib.image import imread

# Učitavanje slike i priprema podataka
img = imread("imgs/test_1.jpg")
img = img.astype(np.float64) / 255  # Normalizacija piksela
w, h, d = img.shape
img_array = np.reshape(img, (w * h, d))  # Transformacija u 2D polje

# Metoda lakta: izračunavanje inercije za različite vrijednosti K
inertia = []
k_values = range(1, 11)  # Testiranje broja grupa od 1 do 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img_array)
    inertia.append(kmeans.inertia_)  # Sprema inerciju za svaki broj klastera

# Grafički prikaz ovisnosti inercije o broju grupa K
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel("Broj grupa (K)")
plt.ylabel("Inercija (J)")
plt.title("Metoda lakta: Ovisnost inercije o broju grupa")
plt.grid(True)
plt.show()
