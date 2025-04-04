import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# ZAD 1 a)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Vizualizacija podataka u x1-x2 ravnini
plt.figure(figsize=(8, 6))

# Trening podaci (točkice)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', label='Trening podaci', edgecolor='k')

# Test podaci (marker 'x')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='x', label='Test podaci')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Prikaz klasifikacijskih podataka (trening i test)')
plt.legend()
plt.grid(True)
plt.show()

# b)

# Kreiranje modela logističke regresije
model = LogisticRegression()

# Treniranje modela na podacima za učenje
model.fit(X_train, y_train)



# c)
# Ekstrakcija parametara modela
intercept = model.intercept_[0]
coefficients = model.coef_[0]

# Generiranje vrijednosti za x1 os
x1_values = np.linspace(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 100)

# Izračun x2 vrijednosti za granicu odluke (theta0 + theta1*x1 + theta2*x2 = 0)
x2_values = -(intercept + coefficients[0] * x1_values) / coefficients[1]

# Vizualizacija
plt.figure(figsize=(8, 6))

# Prikaz trening podataka
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr',
            edgecolor='k', label='Trening podaci')

# Prikaz granice odluke
plt.plot(x1_values, x2_values, color='green', linewidth=2,
         linestyle='--', label='Granica odluke')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistička regresija - granica odluke')
plt.legend()
plt.grid(True)
plt.show()


# d)


# Predikcija na testnim podacima
y_pred = model.predict(X_test)

# Izračun metrika
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Ispis rezultata
print("Matrica zabune:")
print(conf_matrix)
print(f"\nTočnost: {accuracy:.3f}")
print(f"Preciznost: {precision:.3f}")
print(f"Odziv: {recall:.3f}")

# e) Vizualizacija testnog skupa s dobro i pogrešno klasificiranim primjerima

# Kreiranje maski za dobro i pogrešno klasificirane primjere
correct_mask = (y_test == y_pred)
incorrect_mask = ~correct_mask

# Vizualizacija
plt.figure(figsize=(8, 6))

# Dobro klasificirani primjeri (zeleno)
plt.scatter(X_test[correct_mask, 0], X_test[correct_mask, 1],
            c='limegreen', s=70, edgecolor='k',
            label='Dobro klasificirani primjeri')

# Pogrešno klasificirani primjeri (crno)
plt.scatter(X_test[incorrect_mask, 0], X_test[incorrect_mask, 1],
            c='black', s=70, marker='X',
            label='Pogrešno klasificirani primjeri')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Rezultati klasifikacije na testnim podacima')
plt.legend()
plt.grid(True)
plt.show()
