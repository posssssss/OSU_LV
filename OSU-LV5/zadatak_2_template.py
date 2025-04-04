import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


# a)

import numpy as np
import matplotlib.pyplot as plt

# Calculate class counts for training and test sets
train_class_counts = np.unique(y_train, return_counts=True)[1]
test_class_counts = np.unique(y_test, return_counts=True)[1]

# Prepare labels and data for plotting
classes = ['Adelie', 'Chinstrap', 'Gentoo']
x = np.arange(len(classes))
width = 0.35

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot training and test set bars
rects1 = ax.bar(x - width/2, train_class_counts, width,
               label='Trening skup', color='#1f77b4')
rects2 = ax.bar(x + width/2, test_class_counts, width,
               label='Testni skup', color='#ff7f0e')

# Add labels and formatting
ax.set_xlabel('Vrsta pingvina')
ax.set_ylabel('Broj primjera')
ax.set_title('Raspodjela primjera po klasama')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()

# b)
from sklearn.linear_model import LogisticRegression

# Provjera da su varijable X_train i y_train definirane (iz prethodnog koda)
print(f"Dimenzije X_train: {X_train.shape}")  # Trebalo bi biti (n_samples, 2)
print(f"Dimenzije y_train: {y_train.shape}")  # Trebalo bi biti (n_samples,)

# Ako je y_train 2D array (zbog originalnog koda), pretvori ga u 1D
if len(y_train.shape) > 1:
    y_train = y_train.ravel()

# Inicijalizacija i treniranje modela
logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
logistic_model.fit(X_train, y_train)

# Prikaz parametara modela
print("\nParametri modela:")
print(f"Intercepti: {logistic_model.intercept_}")
print(f"Koeficijenti: {logistic_model.coef_}")

# c) Pronalaženje parametara modela
intercepti = logistic_model.intercept_  # Intercepti za svaku klasu
koeficijenti = logistic_model.coef_     # Koeficijenti za svaku klasu

# Ispis intercepta
print("Intercepti (θ₀) za svaku klasu:")
for idx, label in enumerate(['Adelie', 'Chinstrap', 'Gentoo']):
    print(f"{label}: {intercepti[idx]:.3f}")

# Ispis koeficijenata
print("\nKoeficijenti (θ₁, θ₂) za svaku klasu:")
for idx, label in enumerate(['Adelie', 'Chinstrap', 'Gentoo']):
    print(f"{label}: {koeficijenti[idx]}")

# d) Vizualizacija područja odluke
# Obavezno prije pokrenuti:
# - Učitavanje podataka i definicija input_variables/output_variable
# - Podjelu na train/test skupove (X_train, y_train)
# - Treniranje modela (logistic_model)

# Definirajte labels prije poziva funkcije
labels = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

# Poziv funkcije za vizualizaciju
plot_decision_regions(X_train, y_train.ravel(), classifier=logistic_model)  # y_train.ravel() za 1D array
plt.xlabel('Duljina kljuna (mm)')
plt.ylabel('Duljina peraje (mm)')
plt.title('Područja odluke logističke regresije (trening podaci)')
plt.legend(loc='upper left')
plt.show()

# e) Evaluacija modela
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Provjera definiranosti varijabli
try:
    # Predikcija na testnim podacima
    y_pred = logistic_model.predict(X_test)

    # Matrica zabune
    cm = confusion_matrix(y_test, y_pred)
    print("Matrica zabune:")
    print(cm)

    # Točnost
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTočnost: {accuracy:.3f}")

    # Detaljni izvještaj
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Adelie', 'Chinstrap', 'Gentoo']))

except NameError as e:
    print(f"Greška: {e}\nProvjerite da li su X_test i y_test definirani!")
except FileNotFoundError:
    print("Greška: Datoteka 'penguins.csv' nije pronađena!")
except Exception as e:
    print(f"Neočekivana greška: {e}")

# f) Rješenje za prošireni model s ispravkom greške

# Korak 1: Ponovno učitavanje podataka s ispravnom obradom kategorickih varijabli
df = pd.read_csv("penguins.csv")

# Korak 2: Obrada podataka
# a) Obrada nedostajućih vrijednosti
df = df.dropna(subset=['sex'])  # Umjesto dropanja cijelog stupca

# b) Kodiranje kategorickih varijabli
df = pd.get_dummies(df, columns=['island', 'sex'], drop_first=True)

# c) Odabir ulaznih varijabli
input_variables = [
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g',
    'island_Dream',        # Sada postoje nakon get_dummies
    'island_Torgersen',    # Sada postoje nakon get_dummies
    'sex_male'             # Sada postoji nakon get_dummies
]

# Korak 3: Provjera dostupnosti stupaca
missing_cols = [col for col in input_variables if col not in df.columns]
if missing_cols:
    raise ValueError(f"Nedostajući stupci: {missing_cols}")

# Korak 4: Priprema podataka
X = df[input_variables].to_numpy()
y = df['species'].to_numpy().ravel()

# Korak 5: Podjela na train/test skupove
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=123,
    stratify=y
)

# Korak 6: Treniranje modela
logistic_model_extended = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000
)
logistic_model_extended.fit(X_train, y_train)

# Korak 7: Evaluacija
y_pred_ext = logistic_model_extended.predict(X_test)
print(classification_report(y_test, y_pred_ext, target_names=labels.values()))
