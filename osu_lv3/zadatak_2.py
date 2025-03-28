import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")

#a)Pomoću histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz.
data['CO2 Emissions (g/km)'].plot(kind = 'hist', color = 'skyblue', edgecolor = 'gray')
plt.title("Diagram CO2 Emissions")
plt.show()



#c)Pomoću kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip goriva. 
#Primjećujete li grubu mjernu pogrešku u podacima?
data.boxplot(column=["Fuel Consumption Hwy (L/100km)"], by="Fuel Type")
plt.show()

