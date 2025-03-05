
radni_sati = float(input("Radni sati: "))
eura_po_satu = float(input("eura/h: "))

ukupno = radni_sati * eura_po_satu

print("Ukupno:", ukupno, "eura")


def total_euro(sati, cijena_po_satu):
    return sati * cijena_po_satu


radni_sati = float(input("Radni sati: "))
eura_po_satu = float(input("eura/h: "))


ukupno = total_euro(radni_sati, eura_po_satu)

print("Ukupno:", ukupno, "eura")
