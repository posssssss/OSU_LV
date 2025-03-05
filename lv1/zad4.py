rijeci = {}

try:
    with open("song.txt") as file:
      
        for redak in file:
          
            redak = redak.lower()
            for znak in ",.?!;:":
                redak = redak.replace(znak, "")
            
            lista_rijeci = redak.split()
            
            for rijec in lista_rijeci:
                if rijec in rijeci:
                    rijeci[rijec] += 1
                else:
                    rijeci[rijec] = 1

    rijeci_jednom = [rijec for rijec, broj in rijeci.items() if broj == 1]
 
    print("Ukupan broj različitih riječi:", len(rijeci))
    print("Broj riječi koje se pojavljuju samo jednom:", len(rijeci_jednom))
    print("Riječi koje se pojavljuju samo jednom:")
    print(", ".join(rijeci_jednom))

except FileNotFoundError:
    print("Greška: Datoteka 'song.txt' nije pronađena.")