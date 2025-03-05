
brojevi = []

while True:
    unos = input("Unesite broj ili 'Done' za kraj: ")
    
   
    if unos.lower() == "done":
        break
    
    try:
        broj = float(unos)
        brojevi.append(broj) 
    except ValueError:
        print("Greška: Molimo unesite ispravan broj ili 'Done' za kraj.")


if brojevi:
    
    print("Brojeva uneseno:", len(brojevi))
    print("Srednja vrijednost:", sum(brojevi) / len(brojevi))
    print("Minimalna vrijednost:", min(brojevi))
    print("Maksimalna vrijednost:", max(brojevi))
    

    brojevi.sort()
    print("Sortirana lista:", brojevi)
else:
    print("Niste unijeli nijedan broj.")        