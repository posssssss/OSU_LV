import pandas as pd
import numpy as np

data = pd.read_csv("data_C02_emission.csv")

#a)Koliko mjerenja sadrži DataFrame
total_measurements = data.shape[0]
print(total_measurements)

#Kojeg je tipa svaka veličina?
info = data.info()
print(info)

#Postoje li izostale ili duplicirane vrijednosti?
print(data.isnull().sum())
print(data.duplicated().sum())
#ne postoje null vrijednosti, nije ih potrebno brisati

#Kategoričke veličine konvertirajte u tip category
data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')

print(data.info())

#b)Koja tri automobila ima najveću odnosno najmanju gradsku potrošnju? Ispišite u terminal: ime proizvođača, model vozila i kolika je gradska potrošnja.
consumption = data.sort_values(by = ['Fuel Consumption City (L/100km)'])
print("Najmanja potrošnja: ")
print(consumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3))
print("Najveća potrošnja: ")
print(consumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']].tail(3))

#c)Koliko vozila ima veličinu motora između 2.5 i 3.5 L? Kolika je prosječna C02 emisija plinova za ova vozila?
motorSize_data = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]

num_vehicles = motorSize_data.shape[0]
average_CO2 = motorSize_data['CO2 Emissions (g/km)'].mean()

print(num_vehicles)
print(average_CO2)

#d)Koliko mjerenja se odnosi na vozila proizvođača Audi? Kolika je prosječna emisija C02 plinova automobila proizvođača Audi koji imaju 4 cilindara?
#d)Koliko mjerenja se odnosi na vozila proizvođača Audi? Kolika je prosječna emisija C02 plinova automobila proizvođača Audi koji imaju 4 cilindara?
num_audi = data[(data.Make == 'Audi')].shape[0]

filtered_audi = data[(data.Make == 'Audi') & (data.Cylinders == 4)]
CO2_avg = filtered_audi['CO2 Emissions (g/km)'].mean()

print(num_audi)
print(CO2_avg)

#e)Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosječna emisija C02 plinova s obzirom na broj cilindara?
cylinder_counts = data['Cylinders'].value_counts().sort_index()
avg_CO2 = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print(cylinder_counts)
print(avg_CO2)

#f)Kolika je prosječna gradska potrošnja u slučaju vozila koja koriste dizel, a kolika za vozila koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
diesel = data[(data['Fuel Type']  == 'D')]
gasoline = data[(data['Fuel Type'] == 'X')]

print(diesel['Fuel Consumption City (L/100km)'].mean())
print(gasoline['Fuel Consumption City (L/100km)'].mean())

print(diesel['Fuel Consumption City (L/100km)'].median())
print(gasoline['Fuel Consumption City (L/100km)'].median())

#h)Koliko ima vozila ima ručni tip mjenjača (bez obzira na broj brzina)?
manual = data[(data['Transmission'].str.startswith('M'))].shape[0]
print(manual)

#g)Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva?
diesel_max = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
sorted_diesel = diesel_max.sort_values(by = "Fuel Consumption City (L/100km)").head(1)

#i)Izračunajte korelaciju između numeričkih veličina. Komentirajte dobiveni rezultat.
correlation = data.corr(numeric_only = True)
print(correlation)