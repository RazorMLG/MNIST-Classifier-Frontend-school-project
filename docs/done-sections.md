# Popunjena poglavlja

## Issue 01 - Naslovna strana i osnovni podaci

Status: needs-review

Repo oslonci:

- `PRD.md`
- `CONTEXT.md`

Rezime upisa:

- Dodat je draft naslovne strane sa placeholder-ima za naziv skole, smer, odeljenje, predmet, mentora, autora, mesto i godinu izrade.
- Upisan je kratak opis rada, kao i apstrakt i predlog kljucnih reci zasnovani na stvarnom obimu MNIST aplikacije iz repozitorijuma.

Otvorene rupe:

- [x] Uneti tacan naziv skole, obrazovni profil, odeljenje, predmet, ime autora i ime mentora.
- [x] Usaglasiti raspored naslovne strane sa zvanicnim skolskim obrascem.
- [x] Po potrebi prilagoditi apstrakt i kljucne reci zahtevima skole ili dodati verziju na stranom jeziku.

## Issue 02 - Uvod

Status: drafted

Repo oslonci:

- `issues/done/issue-1-bootable-app-shell/text/issue.md`
- `issues/done/issue-2-first-built-in-model-slice/text/issue.md`
- `PRD.md`
- `CONTEXT.md`
- `backend/app/main.py`
- `frontend/src/App.tsx`
- `backend/tests/test_custom_training.py`
- `frontend/src/App.test.tsx`

Rezime upisa:

- Dodat je uvod koji definise problem klasifikacije rukom pisanih cifara i obrazlaze zasto je MNIST pogodan kao standardizovan nastavni i demonstracioni primer.
- U tekst je ukljuceno objasnjenje da projekat nije samo pojedinacni model, vec aplikaciona celina sa backend-om za predikciju i trening i frontend-om za crtanje cifre, izbor modela i pregled rezultata.
- Poglavlje je zavrseno prelazom ka narednoj celini u kojoj se formalno odredjuju predmet, cilj i zadaci rada.

Otvorene rupe:

- [x] Pri finalnoj redakturi proveriti da li terminologiju o modelima i katalogu treba dodatno stilizovati prema ostatku rada.

## Issue 03 - Predmet, cilj i zadaci rada

Status: drafted

Repo oslonci:

- `PRD.md`
- `docs/README.md`
- `start.ps1`
- `package.json`
- `backend/app/main.py`
- `frontend/src/App.tsx`
- `frontend/src/App.test.tsx`
- `CONTEXT.md`

Rezime upisa:

- Formalno je definisan predmet rada kao lokalno pokretljiv web sistem za klasifikaciju MNIST cifara koji objedinjuje Python backend i React korisnicki interfejs.
- Razdvojeni su opsti, istrazivacki i implementacioni cilj rada, sa naglaskom na objedinjavanje vise modela, interaktivnu predikciju i prilagodjeno treniranje.
- Dodata je numerisana lista konkretnih zadataka rada zasnovana na stvarnim funkcionalnostima repozitorijuma: root startup, health provera, katalog modela, predikcija sa platna, CSV preview, trening tok i osnovna verifikacija.

Otvorene rupe:

- [ ] Pri finalnoj redakturi proveriti da li skola zahteva drugaciji formalni raspored cilja i zadataka rada.

## Issue 04 - Teorijska osnova

Status: needs-literature

Repo oslonci:

- `CONTEXT.md`
- `backend/app/reference_model.py`
- `backend/app/shipped_classical_training.py`
- `backend/app/shipped_deep_training.py`
- `backend/app/shipped_classical_models.py`
- `backend/app/custom_training.py`

Rezime upisa:

- Dodat je formalni teorijski pregled MNIST skupa podataka, nadgledanog viseklasnog ucenja i znacaja generalizacije pri evaluaciji klasifikatora.
- Upisano je prosireno objasnjenje preprocessing-a koje je vezano za stvarnu implementaciju projekta: normalizacija, skaliranje na 28x28 format, centriranje prema centru mase i PCA redukcija dimenzionalnosti za klasicne modele.
- Poglavlje je razlozeno na zasebne podsekcije za referentni prototipski pristup, k-NN, linearnu SVM, Random Forest, MLP i CNN, uz dublje implementaciono obrazlozenje tamo gde repo direktno potvrduje hiperparametre i arhitekturu.

Otvorene rupe:

- [ ] Dopuniti bibliografske izvore za MNIST, nadgledano ucenje, odabrane algoritme i metrike evaluacije.
- [ ] Odluciti da li u finalnoj verziji rada treba dodati sliku preprocessing toka i tabelarni uporedni pregled porodica modela.

## Issue 05 - Tehnologije i razvojno okruzenje

Status: drafted

Repo oslonci:

- `requirements.txt`
- `package.json`
- `frontend/package.json`
- `start.ps1`
- `backend/app/main.py`
- `docs/README.md`

Rezime upisa:

- Dodat je formalni pregled backend i frontend tehnoloskog steka, sa objasnjenjem uloge Python-a, FastAPI-ja, Pydantic validacije, React-a, TypeScript-a i Vite okruzenja.
- Objasnjena je podela biblioteka za masinsko ucenje na scikit-learn za klasicne modele i PyTorch za duboke modele, kao i uloga pytest, vitest i pomocnih razvojnih alata.
- Upisan je konkretan tok lokalnog pokretanja preko `start.ps1`, ukljucujuci kreiranje `.venv` okruzenja, instalaciju zavisnosti i paralelno podizanje backend i frontend servera.

Otvorene rupe:

- [ ] Dopuniti tacne verzije Python, Node.js i PowerShell okruzenja ako ih skolski obrazac zahteva eksplicitno.
- [ ] Dopuniti formalne hardverske pretpostavke za lokalno pokretanje, posto nisu dokumentovane u repozitorijumu.

## Issue 06 - Arhitektura sistema

Status: drafted

Repo oslonci:

- `issues/done/issue-1-bootable-app-shell/text/issue.md`
- `issues/done/issue-5-first-custom-training-slice/text/issue.md`
- `PRD.md`
- `CONTEXT.md`
- `start.ps1`
- `backend/app/main.py`
- `backend/app/custom_training.py`
- `frontend/src/App.tsx`
- `data/registry/models.json`
- `backend/tests/test_health.py`
- `backend/tests/test_custom_training.py`

Rezime upisa:

- Dodat je formalni opis sistema kao modularnog monolita sa tri jasno odvojena sloja: React frontend, FastAPI backend i lokalni storage za modele i registar.
- Objasnjena je uloga startup skripte, backend lifecycle inicijalizacije, zajednickog model kataloga i model registry mehanizma koji povezuje shipped i custom modele.
- Upisan je tok predikcije i tok asinhronog treninga od korisnicke akcije do odgovora backend-a i osvezavanja zajednickog kataloga na frontend strani.

Otvorene rupe:

- [ ] Dopuniti arhitektonski dijagram sistema ili screenshot dijagrama koji prati opis slojeva i tokova.

## Issue 07 - Backend implementacija

Status: drafted

Repo oslonci:

- `issues/done/issue-2-first-built-in-model-slice/text/issue.md`
- `issues/done/issue-4-csv-intake-and-split-preview/text/issue.md`
- `issues/done/issue-5-first-custom-training-slice/text/issue.md`
- `issues/done/issue-6-shipped-classical-model-expansion/text/issue.md`
- `issues/done/issue-7-shipped-deep-model-expansion/text/issue.md`
- `PRD.md`
- `CONTEXT.md`
- `backend/app/main.py`
- `backend/app/reference_model.py`
- `backend/app/training_csv.py`
- `backend/app/custom_training.py`
- `backend/app/shipped_classical_models.py`
- `backend/app/shipped_classical_training.py`
- `backend/app/shipped_deep_training.py`
- `backend/tests/test_health.py`
- `backend/tests/test_training_csv_preview.py`
- `backend/tests/test_custom_training.py`
- `backend/tests/test_shipped_classical_training.py`
- `backend/tests/test_shipped_deep_training.py`

Rezime upisa:

- Dodat je tehnicki pregled FastAPI API sloja, Pydantic validacije i backend ruta za health, models, predict, CSV preview i trening poslove.
- Objasnjen je kanonski preprocessing platna, CSV intake i zajednicki model lifecycle koji povezuje referentne, shipped i custom modele kroz isti katalog i isti inferencioni ugovor.
- Upisan je opis asinhronog custom trening toka, persistencije metapodataka i runtime artefakata, kao i testova koji verifikuju backend ugovore.

Otvorene rupe:

- [ ] Po potrebi dopuniti prilog sa jednim ili dva konkretna primera JSON zahteva i odgovora za kljucne backend rute.

## Issue 08 - Frontend pregled

Status: drafted

Repo oslonci:

- `issues/done/issue-1-bootable-app-shell/text/issue.md`
- `issues/done/issue-3-shipped-model-leaderboard-and-details/text/issue.md`
- `issues/done/issue-4-csv-intake-and-split-preview/text/issue.md`
- `issues/done/issue-5-first-custom-training-slice/text/issue.md`
- `issues/done/issue-8-custom-classical-training-expansion/text/issue.md`
- `issues/done/issue-9-custom-deep-training-expansion/text/issue.md`
- `issues/done/issue-10-version-one-policy-cleanup/text/issue.md`
- `docs/README.md`
- `frontend/src/App.tsx`
- `frontend/src/App.css`
- `frontend/src/App.test.tsx`

Rezime upisa:

- Dodat je akademski pregled frontend sloja kao prezentacionog i orkestracionog dela sistema, sa objasnjenjem bootstrap faze, izbora modela i odnosa prema backend API-ju.
- Upisan je opis dva radna rezima, pri cemu prediction koristi 20x20 canvas i prikaz raspodele pouzdanosti, a training uvodi CSV upload, preview podele skupa, kurirane hiperparametre, warning poruke i pracenje asinhronog trening posla.
- Objasnjen je zajednicki leaderboard i detalji modela, ukljucujuci tabbed deep details, training snapshot za custom modele, pravilo rucnog imenovanja i inline obradu neuspelih treninga, uz osvrt na frontend testove koji verifikuju glavne korisnicke tokove.

Otvorene rupe:

- [ ] Dodati jedan ili dva screenshot-a interfejsa koji odgovaraju placeholder-ima ostavljenim u poglavlju.

## Issue 09 - Faze implementacije

Status: drafted

Repo oslonci:

- `issues/done/issue-1-bootable-app-shell/text/issue.md`
- `issues/done/issue-2-first-built-in-model-slice/text/issue.md`
- `issues/done/issue-3-shipped-model-leaderboard-and-details/text/issue.md`
- `issues/done/issue-4-csv-intake-and-split-preview/text/issue.md`
- `issues/done/issue-5-first-custom-training-slice/text/issue.md`
- `issues/done/issue-6-shipped-classical-model-expansion/text/issue.md`
- `issues/done/issue-7-shipped-deep-model-expansion/text/issue.md`
- `issues/done/issue-8-custom-classical-training-expansion/text/issue.md`
- `issues/done/issue-9-custom-deep-training-expansion/text/issue.md`
- `issues/done/issue-10-version-one-policy-cleanup/text/issue.md`
- `PRD.md`
- `CONTEXT.md`
- `docs/README.md`
- `backend/app/custom_training.py`
- `backend/tests/test_custom_training.py`
- `frontend/src/App.tsx`
- `frontend/src/App.test.tsx`

Rezime upisa:

- Skeleton sekcija je pretvorena u formalno poglavlje koje rekonstruise razvoj projekta kroz deset zavrsenih razvojnih faza, od bootable shell-a do zavrsnog uskladjivanja version-one pravila.
- Svaka faza je opisana kroz problem koji je resavala, uvedeno tehnicko resenje i neposredan rezultat, uz eksplicitnu vezu izmedju planerskih odluka iz `PRD.md` i realizovanih inkremenata u kodu.
- Posebno je izdvojeno kako su zajednicki Model Catalog, asinhroni Training Job lifecycle i deep-details obrazac postepeno prosirivani sa shipped na custom modele, ukljucujuci custom deep training slice iz razvojnog issue-a broj 9.

Otvorene rupe:

- [ ] Ako se naknadno potvrde datumi commit-ova, prezentacionih milestone-a ili skolski zahtev za vremenskom linijom, dopuniti poglavlje tabelom ili hronoloskim prilogom bez uvodjenja neproverenih podataka.

## Issue 10 - Testiranje i verifikacija

Status: drafted

Repo oslonci:

- `PRD.md`
- `package.json`
- `backend/tests/test_health.py`
- `backend/tests/test_reference_prediction.py`
- `backend/tests/test_training_csv_preview.py`
- `backend/tests/test_custom_training.py`
- `backend/tests/test_shipped_classical_training.py`
- `backend/tests/test_shipped_deep_training.py`
- `frontend/src/App.test.tsx`

Rezime upisa:

- Dodat je formalni pregled test strategije sa naglaskom na spolja vidljive ugovore, korisnicke tokove i version-one pravila, u skladu sa testnim smernicama iz `PRD.md`.
- Upisan je pregled backend testova za health, predikciju, CSV preview, custom training lifecycle, pravilo jednog aktivnog treninga i regeneraciju shipped modela.
- Dodat je opis frontend scenarija za bootstrap, prediction, leaderboard i detalje modela, training mode, CSV validaciju i policy pravila rucnog imenovanja i inline obrade neuspelog treninga, uz navodjenje komandi `npm run test` i `npm run typecheck`.

Otvorene rupe:

- [ ] Repo ne sadrzi formalni coverage izvestaj ni trajno sacuvan zbirni benchmark, pa takve metrike ne unositi bez dodatnog merenja.
- [ ] Ako skolska forma trazi tabelarni pregled test slucajeva ili konkretne izlaze komandi, dopuniti ih zasebnim prilogom zasnovanim na stvarnom pokretanju suite-a.

## Issue 11 - Zakljucak

Status: drafted

Repo oslonci:

- `PRD.md`
- `docs/README.md`
- `issues/done/issue-10-version-one-policy-cleanup/text/issue.md`
- `CONTEXT.md`

Rezime upisa:

- Dodat je zavrsni akademski zakljucak koji sumira ostvareni sistem kao lokalno pokretljivu aplikacionu celinu sa backend-om, frontend-om, zajednickim katalogom modela i podrskom za predikciju i prilagodjeno treniranje.
- U tekst je ukljuceno poredjenje planiranog i realizovanog obima rada sa naglaskom na dosledno sprovedena version-one pravila: jedan aktivan trening, rucno imenovanje custom modela, inline obrada neuspelih treninga i jedinstvena startup tacka.
- Izdvojena su ogranicenja prve verzije i cetiri konkretna pravca buduceg razvoja koji su utemeljeni u trenutno vanopseznim ili svesno odlozenim mogucnostima projekta.

Otvorene rupe:

- [ ] Pri finalnoj redakturi proveriti da li skola ocekuje jos licniji ton zavrsnog poglavlja ili je zadrzani formalno-akademski stil odgovarajuci.

## Issue 12 - Literatura

Status: needs-literature

Repo oslonci:

- `PRD.md`
- `CONTEXT.md`
- `docs/README.md`
- `requirements.txt`

Rezime upisa:

- Skeleton sekcija je pretvorena u formalno poglavlje koje razdvaja strucnu literaturu, izvor podataka i tehnicku dokumentaciju koriscenu u radu.
- U tekst su eksplicitno uvedeni knjiga Sebastiana Raschke i saradnika o PyTorch i scikit-learn pristupu, knjiga _Deep Learning with PyTorch_ i Kaggle kao izvor MNIST-kompatibilnog CSV skupa.
- Dodat je metodoloski pasus koji objasnjava da se interni projektni dokumenti koriste kao pomocni izvori za terminologiju i opseg projekta, ali ne kao spoljna naucna literatura.

Otvorene rupe:

- [ ] Uneti potpune bibliografske podatke za knjigu Sebastiana Raschke i saradnika u formatu koji skola zahteva.
- [ ] Uneti potpune bibliografske podatke za knjigu _Deep Learning with PyTorch_, ukljucujuci izdanje, godinu i izdavaca.
- [ ] Upisati tacan naziv Kaggle skupa, URL i datum pristupa.
- [ ] Po potrebi dopuniti zvanicne URL-ove tehnicke dokumentacije i proveriti da li ih skolski obrazac svrstava u zasebnu webografiju.
