try:    
    ocjena = float(input("broj između 0.0 i 1.0: "))
    if ocjena < 0.0 or ocjena > 1.0:
            print("Unesite ocjenu u razmaku od 0.0 do 1.0")

    else: 
        if ocjena >= 0.9:
            print("ocjena A")
        elif ocjena >= 0.8:
            print("ocjena B") 
        elif ocjena >= 0.7:
            print("ocjena C") 
        elif ocjena >= 0.6:
            print("ocjena D") 
        elif ocjena < 0.6:
            print("ocjena F") 

except ValueError:
    print("Greška: Unos mora biti broj između 0.0 i 1.0.")