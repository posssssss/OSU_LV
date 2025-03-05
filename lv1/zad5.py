
ham_broj_rijeci = 0
ham_broj_poruka = 0
spam_broj_rijeci = 0
spam_broj_poruka = 0
spam_zavrsava_usklicnikom = 0


try:
    with open('SMSSpamCollection.txt') as fhand:
        for linija in fhand:
            linija = linija.strip()
            if linija:
             
                tip, tekst = linija.split('\t', 1)
                rijeci = tekst.split()
                broj_rijeci = len(rijeci)
                
                if tip == 'ham':
                    ham_broj_rijeci += broj_rijeci
                    ham_broj_poruka += 1
                elif tip == 'spam':
                    spam_broj_rijeci += broj_rijeci
                    spam_broj_poruka += 1
                 
                    if tekst.endswith('!'):
                        spam_zavrsava_usklicnikom += 1


    prosjek_ham = ham_broj_rijeci / ham_broj_poruka if ham_broj_poruka > 0 else 0
    prosjek_spam = spam_broj_rijeci / spam_broj_poruka if spam_broj_poruka > 0 else 0

 
    print(f"Prosječan broj riječi u 'ham' porukama: {prosjek_ham:.2f}")
    print(f"Prosječan broj riječi u 'spam' porukama: {prosjek_spam:.2f}")
    print(f"Broj 'spam' poruka koje završavaju uskličnikom: {spam_zavrsava_usklicnikom}")

except FileNotFoundError:
    print("Greška: Datoteka 'SMSSpamCollection.txt' nije pronađena.")
