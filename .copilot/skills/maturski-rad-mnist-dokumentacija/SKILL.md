---
name: maturski-rad-mnist-dokumentacija
description: This skill generates thesis-ready documentation for the MNIST classifier project by reconstructing implementation phases from issues, code, tests, and existing docs. Use when the user asks for a maturski rad, project documentation, backend-focused architecture text, or implementation phases derived from this repository.
---

# Dokumentacija Za Maturski Rad O MNIST Projektu

## Osnovna pravila

- Podrazumevani jezik izlaza je srpski, osim ako korisnik izricito ne trazi drugi jezik.
- Cinjenice se utvrdjuju ovim redosledom: `issues/done`, `PRD.md`, `CONTEXT.md`, `backend/app`, `backend/tests`, `docs/README.md`, `start.ps1`, pa zatim `frontend/src/App.tsx`.
- Pri opisu faza implementacije svaku fazu prvo potvrditi u kodu i testovima, a tek potom pretvoriti u zavrsen tekst za rad.
- Pri pisanju arhitekture teziste drzati na backend delu sistema, dok frontend treba obraditi krace, osim ako korisnik izricito ne zatrazi siri prikaz.

Primer zahteva:

`Napravi poglavlje o fazama implementacije za maturski rad na osnovu issues/done i objasni backend arhitekturu uz kratak osvrt na frontend.`

## Tok rada

### 1. Rekonstrukcija faza implementacije

1. Procitati `issues/done/issue-*/text/issue.md` redom po broju.
2. Za svaku stavku povezati planiranu funkcionalnost sa konkretnim fajlovima, endpointima, modulima, izmenama u interfejsu i testovima.
3. Svaku fazu prikazati kroz: cilj, glavne izmene, backend fokus, frontend dopunu, verifikaciju i korisnicki rezultat.
4. Hronologiju razvoja prikazati jasno i navesti zavisnosti medju fazama kada su vidljive iz issue dokumentacije.

### 2. Izrada poglavlja rada

1. Cinjenice iz repozitorijuma pretvoriti u formalnu srpsku prozu, bez prepisivanja issue skracenica i radnih formulacija.
2. Kada korisnik trazi celovit nacrt rada, preporuceni redosled poglavlja je: uvod i cilj, tehnologije, teorijska osnova izabranih klasifikatora, arhitektura sistema, backend implementacija, pregled frontenda, faze implementacije, testiranje i verifikacija, zakljucak.
3. U arhitektonskom delu najpre objasniti FastAPI rute, orkestraciju treninga, cuvanje modela, odnos shipped i custom modela, unos CSV podataka i tok predikcije, a tek zatim krace opisati UI deo.
4. Svaku opisanu fazu osloniti najmanje na jedan issue i jedan implementacioni fajl.
5. Ostaviti posebno poglavlje ili potpoglavlje u kome korisnik sam objasnjava teoriju masinskog ucenja iza izabranih klasifikatora; taj prostor ne popunjavati detaljnim teorijskim tekstom osim ako korisnik to izricito ne zatrazi.

### 3. Objasnjenje arhitekture sa naglaskom na backend

1. `backend/app/main.py` tretirati kao glavno ulazno mesto API-ja i pregled raspolozivih ruta.
2. Fajlove `custom_training.py`, `training_csv.py`, `reference_model.py`, `shipped_classical_models.py`, `shipped_deep_training.py` i `shipped_model_training.py` koristiti za objasnjenje backend odgovornosti.
3. Na `backend/tests` se oslanjati kada treba pokazati kako su verifikovani tokovi rada i stabilnost ponasanja.
4. Frontend pomenuti pre svega kao React + TypeScript interfejs za crtanje cifre, izbor modela, prikaz detalja i pokretanje treninga.

## Pravila izlaza

- Ne izmisljati faze implementacije, metrike, datume ni odluke koje nisu potvrdjene u repozitorijumu.
- `issues/done` tretirati kao razvojni trag, ali svaku tvrdnju proveriti u aktuelnom kodu.
- `PRD.md` i `CONTEXT.md` koristiti za terminologiju, granice sistema i produktni kontekst.
- Pri opisu pokretanja projekta obavezno navesti `start.ps1` kao vidljivu Windows ulaznu tacku.
- Kada dokazi nisu potpuni, tvrdnju oznaciti kao zakljucak na osnovu dostupnih tragova ili kao otvoreno pitanje, a ne kao potvrdenu cinjenicu.
- Pregled frontenda zadrzati kracim od backend dela, osim ako korisnik izricito ne trazi ravnomernu obradu.
- Posebno cuvati prostor za teorijsko objasnjenje klasifikatora i podrazumevano ga ostaviti kao okvir ili kratak uvod, umesto da bude popunjen kompletnim ML objasnjenjem.
- Prednost dati jasnoj, sazetoj i formalnoj prozi primerenoj maturskom radu, osim kada korisnik trazi outline ili radnu skicu.
