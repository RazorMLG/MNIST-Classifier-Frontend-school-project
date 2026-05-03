# Klasifikator za MNIST podatke koriscenjem masinskog ucenja

<!-- markdownlint-disable MD024 -->

## Issue 01 - Naslovna strana i osnovni podaci

### Naslovna strana

Srednja Skola "Sveti Trifun sa domom učenika"
Aleksandrovac

MATURSKI RAD

Predmet: Objektivno orijentisano programiranje

Tema: Klasifikator za MNIST podatke koriscenjem masinskog ucenja

Mentor: Saša Knežević
Ucenik: Jovan Prsić

Aleksandrovac, maj 2026.

### Osnovni podaci o radu

Rad pod naslovom "Klasifikator za MNIST podatke koriscenjem masinskog ucenja" bavi se razvojem softverskog sistema za klasifikaciju rukom pisanih cifara nad MNIST skupom podataka. U okviru projekta objedinjeni su Python backend zaduzen za obradu podataka, treniranje i predikciju, kao i React korisnicki interfejs namenjen prikazu modela, unosu cifre preko platna i pokretanju prilagodjenog treniranja nad MNIST-kompatibilnim CSV skupovima podataka. Administrativni elementi naslovne strane dopunjavaju se prema zvanicnom obrascu skole.

### Apstrakt

Ovaj maturski rad prikazuje izradu aplikacije za klasifikaciju rukom pisanih cifara zasnovane na MNIST skupu podataka. Sistem je realizovan kao celina koju cine web korisnicki interfejs i serverska aplikacija za upravljanje modelima, obradu ulaznih podataka, izvrsavanje predikcije i pokretanje procesa treniranja. Posebna paznja posvecena je uporednom prikazu vise porodica modela masinskog ucenja, ukljucujuci klasicne pristupe i duboke neuronske mreze, kao i mogucnosti prilagodjenog treniranja nad korisnicki dostavljenim podacima. Rad ima za cilj da prikaze kako se teorijski koncepti klasifikacije slika mogu povezati sa konkretnom implementacijom aplikacije pogodnom za demonstraciju, eksperimentisanje i obrazovne potrebe.

### Kljucne reci

MNIST, klasifikacija slika, masinsko ucenje, neuronske mreze, web aplikacija, Python, React

## Issue 02 - Uvod

Razvoj sistema za automatsko prepoznavanje rukom pisanih cifara predstavlja jedan od najpoznatijih primera primene masinskog ucenja na problem klasifikacije slika. Zadatak se sastoji u tome da se na osnovu ulazne slike odredi kojoj od deset mogucih cifara pripada posmatrani zapis. Iako je na prvi pogled rec o usko definisanom klasifikacionom problemu, njegova prakticna realizacija podrazumeva vise od samog izbora algoritma: potrebno je obezbediti doslednu obradu ulaza, jasan nacin uporedjivanja modela i okruzenje u kome se rezultat moze neposredno proveriti kroz rad gotove aplikacije.

MNIST skup podataka je pogodan za ovakav tip rada zato sto predstavlja standardizovan i siroko prihvacen primer za uvod u klasifikaciju rukom pisanih cifara. Njegova vrednost u nastavnom i demonstracionom kontekstu ogleda se u tome sto omogucava da se na razumljivom problemu povezu teorijski pojmovi iz oblasti masinskog ucenja, obrade slike i evaluacije modela sa konkretnom implementacijom koja moze da se pokrene na uobicajenoj razvojnoj masini. Zbog toga MNIST nije koristan samo kao izvor podataka za treniranje, vec i kao stabilna osnova za poredjenje razlicitih pristupa i za prikaz nacina na koji se modeli uklapaju u siri softverski sistem.

U ovom projektu teziste nije postavljeno samo na izradu jednog klasifikatora, vec na razvoj celovite aplikacije za rad sa MNIST podacima. Postojeca implementacija obuhvata Python backend koji proverava spremnost sistema, ucitava dostupne modele, prima ulaz nacrtane cifre, izvrsava predikciju i podrzava tokove prilagodjenog treniranja nad MNIST-kompatibilnim CSV skupovima podataka. Uz to, React korisnicki interfejs omogucava crtanje cifre, izbor modela, pregled osnovnih metapodataka i pokretanje treninga kroz jedinstveno radno okruzenje. Time projekat pokazuje da uspesna primena masinskog ucenja ne zavisi samo od kvaliteta modela, vec i od nacina na koji su obrada podataka, korisnicka interakcija i organizacija artefakata povezani u funkcionalnu celinu.

Cilj aplikacije je da korisniku omoguci interaktivnu predikciju nad nacrtanom cifrom, pregled i poredjenje vise modela u zajednickom katalogu, kao i eksperimentisanje sa prilagodjenim treniranjem bez napustanja iste aplikacije. Takav pristup radu ima dvostruku vrednost: sa jedne strane omogucava demonstraciju osnovnih koncepata klasifikacije slika, a sa druge strane prikazuje kako se teorijski model pretvara u upotrebljiv i pregledno organizovan softverski proizvod. Polazeci od ovako definisanog problema i obima sistema, u narednom poglavlju bice formalno odredjeni predmet, cilj i konkretni zadaci rada.

## Issue 03 - Predmet, cilj i zadaci rada

### Predmet rada

Predmet ovog rada jeste projektovanje i realizacija lokalno pokretljivog web sistema za klasifikaciju rukom pisanih cifara zasnovanog na MNIST skupu podataka. Sistem je koncipiran kao celina koju cine Python backend za proveru spremnosti okruzenja, upravljanje katalogom modela, obradu ulaznih podataka, predikciju i trening, kao i React korisnicki interfejs namenjen interaktivnom radu sa modelima.

### Cilj rada

Opsti cilj rada jeste da se teorijski koncept klasifikacije slika poveze sa konkretnom softverskom implementacijom koja omogucava pregledno poredjenje vise porodica modela masinskog ucenja u istom aplikacionom okruzenju. U istrazivackom smislu cilj je da se na standardizovanom problemu MNIST klasifikacije prikaze kako razliciti modeli mogu biti objedinjeni u zajednicki katalog i uporedivi kroz iste tokove rada, dok je implementacioni cilj izrada aplikacije koja se pokrece iz korena repozitorijuma, omogucava predikciju nad cifrom nacrtanom na platnu, prikazuje metapodatke o raspolozivim modelima i podrzava prilagodjeno treniranje nad MNIST-kompatibilnim CSV skupovima podataka.

### Zadaci rada

Radi ostvarenja navedenog cilja, u radu su definisani sledeci zadaci:

1. Obezbediti jedinstven nacin lokalnog pokretanja sistema preko korenskog PowerShell skripta `start.ps1`, uz pripremu Python okruzenja, instalaciju zavisnosti i pokretanje backend i frontend dela aplikacije.
2. Realizovati serverski sloj koji proverava spremnost sistema i izlaže osnovne API funkcionalnosti za proveru zdravlja servisa, ucitavanje kataloga modela, izvrsavanje predikcije, pregled CSV podele skupa i pokretanje trening poslova.
3. Omoguciti korisniku da kroz graficki interfejs izabere model iz zajednickog kataloga i izvrsi predikciju nad cifrom nacrtanom na platnu, uz prikaz rezultata klasifikacije i odgovarajucih vrednosti pouzdanosti.
4. Obezbediti pregled modela u okviru jedinstvenog kataloga, sa naglaskom na uporedni prikaz isporucenih i korisnicki istreniranih modela, njihovih metrika i osnovnih metapodataka potrebnih za tumacenje rezultata.
5. Implementirati unos MNIST-kompatibilnih CSV skupova podataka, validaciju formata i prikaz podele na trening, validacioni i test skup pre pokretanja obuke.
6. Realizovati tok prilagodjenog treniranja koji prihvata naziv modela, izbor porodice modela, seme slucajnosti i kurirane hiperparametre, a zatim vodi i prati asinhroni trening posao.
7. Obezbediti proveru ispravnosti sistema kroz eksplicitnu proveru stanja backend-a pri pokretanju aplikacije i kroz automatizovane testove koji potvrduju rad kljucnih backend i frontend tokova.

## Issue 04 - Teorijska osnova

### MNIST kao standardni skup podataka

MNIST (Modified National Institute of Standards and Technology) predstavlja standardni skup za uvodne i uporedne eksperimente u klasifikaciji rukom pisanih cifara. Svaki primer sastoji se od slike dimenzija 28x28 piksela u sivim tonovima i pridruzene oznake iz skupa cifara od 0 do 9. Zbog jednoobraznog formata i velike zastupljenosti u literaturi, MNIST omogucava da se razliciti algoritmi porede nad istim problemom i u istom reprezentacionom prostoru. U prakticnom smislu to znaci da se razlika izmedju modela ne zasniva na promeni tipa ulaza, vec na nacinu na koji model iz istog vektora osobina izvlaci diskriminativne obrasce. [Literatura dopuniti: osnovni izvor za MNIST i pregledni rad o klasifikaciji rukom pisanih cifara.]

U ovom projektu MNIST ima i dodatnu metodolosku ulogu, jer povezuje dve vrste ulaza: standardne redove iz CSV skupa za treniranje i cifru koju korisnik crta na platnu u okviru web aplikacije. Da bi poredjenje modela bilo smisleno, oba ulaza se na backend strani svode na uskladjenu 28x28 reprezentaciju pogodnu za dalje bodovanje, evaluaciju i cuvanje metapodataka o modelima.

### Nadgledano masinsko ucenje i zadatak klasifikacije

Klasifikacija MNIST cifara pripada oblasti nadgledanog masinskog ucenja. Kod ovakvog pristupa raspolaze se skupom primera u kome je svakom ulaznom zapisu pridruzena tacna klasa, a cilj algoritma je da na osnovu tih primera nauci funkciju koja i za nove, ranije nevidjene ulaze odredjuje najverovatniju oznaku. U posmatranom problemu izlazni prostor je diskretan i obuhvata deset klasa, sto zadatak svrstava u viseklasnu klasifikaciju. [Literatura dopuniti: uvodni izvor za nadgledano ucenje i klasifikaciju.]

Sa teorijskog stanovista kvalitet klasifikatora ne procenjuje se samo po tome da li je na naucenim primerima dao tacan odgovor, vec pre svega po sposobnosti generalizacije. Zbog toga se u praksi skup podataka deli na deo za ucenje parametara modela i deo za proveru uspesnosti. Upravo taj princip je prisutan i u ovom projektu: treniranje i evaluacija shipped i custom modela zasnivaju se na podeli skupa na trening, validacioni i test deo, cime se omogucava odvojeno pracenje procesa obuke i zavrsne procene modela.

### Znacaj preprocessing-a ulaza

Kod klasifikacije slika priprema podataka ima veliku vaznost zato sto isti broj piksela ne garantuje i istu geometriju zapisa. Rukom napisana cifra moze biti pomerena, tanja ili deblja, sa razlicitim odnosom praznog prostora oko centralnog sadrzaja. Ako se takve varijacije ne ublaze, model moze uciti nepozeljne razlike koje nisu vezane za samu klasu, vec za poziciju ili razmeru zapisa.

Implementacija u ovom projektu eksplicitno uvodi takvo uskladjivanje ulaza. Kod cifre nacrtane na platnu backend najpre pronalazi ogranicavajuci okvir aktivnih piksela, zatim izdvojeni sadrzaj skalira tako da stane u sredisnji okvir od 20x20 unutar ciljne slike 28x28 i na kraju ga translira prema centru mase. Kod MNIST redova iz CSV formata svaka instanca vec sadrzi 784 vrednosti piksela, ali se i tu vrsi normalizacija intenziteta i ponovno centriranje prema centru mase. Na taj nacin se i korisnicki crtani ulaz i zapisi iz skupa dovode u sto slicniji reprezentacioni prostor, sto je posebno vazno za modele koji direktno porede raspored intenziteta po pikselima.

#### Normalizacija, izdvajanje sadrzaja i centriranje

Prvi korak preprocessing-a jeste svodjenje ulaza na uporediv raspon vrednosti i oblik. Kod canvas ulaza intenziteti se prevode na standardizovane realne vrednosti, zatim se pronalazi region u kome zaista postoji sadrzaj, dok se prazna margina zanemaruje. Time se klasifikatoru ne prepusta da sam zakljucuje da li je neki zapis vise ulevo, udesno ili blize gornjoj ivici, vec se naglasak vraca na raspored poteza koji cine samu cifru.

Centriranje prema centru mase ima dodatni znacaj zato sto pomeraj cifre za nekoliko piksela moze izrazito promeniti sirovi vektor osobina iako vizuelno nije promenjena njena klasa. Kada se sadrzaj prevede u standardni koordinatni okvir, modeli lakse uce pravilnosti koje su vezane za oblik cifre, a ne za slucajnu poziciju crtanja. U teorijskom smislu ovo je primer transformacije koja smanjuje varijansu ulaza koja nije semanticki vazna za zadatak klasifikacije. [Slika dopuniti: primer toka obrade od canvas cifre ili CSV reda do standardizovane 28x28 matrice.]

#### PCA kao redukcija dimenzionalnosti

PCA (principal component analysis) nije klasifikator, vec linearna transformacija kojom se veliki broj medjusobno povezanih osobina prevodi u manji broj novih koordinata. Te nove koordinate, odnosno glavne komponente, biraju se tako da zadrze sto veci deo varijanse prisutne u originalnim podacima. Kod MNIST zapisa to je vazno zato sto 784 piksela nose mnogo redundantnih informacija: susedni pikseli su cesto povezani, a deo prostora ostaje prazan kod velikog broja primera. [Literatura dopuniti: izvor za PCA i redukciju dimenzionalnosti.]

U projektu PCA prethodi klasicnim shipped modelima kao deo zajednickog sklearn Pipeline lanca. Kod k-NN koristi se projekcija na 32 komponente, dok se za linearnu SVM i Random Forest koristi 48 komponenti. Time se postižu dve prakticne koristi. Prva je smanjenje racunske slozenosti, jer klasifikatori ne rade nad punim 784-dimenzionalnim prostorom. Druga je delimicno uklanjanje suma i redundantnosti, pa modeli donose odluku nad sazetijom reprezentacijom. Istovremeno, PCA uvodi i kompromis: deo informacije se neminovno gubi, pa izbor broja komponenti predstavlja ravnotezu izmedju kompaktnosti i ocuvanja diskriminativnih obrazaca.

### Klasicni i duboki modeli u projektu

U teoriji klasifikacije korisno je razlikovati modele koji odluku zasnivaju na geometriji i inzenjerski pripremljenim osobinama od modela koji obrasce uce kroz vise slojeva nelinearnih transformacija. Obe grupe su prisutne u ovom projektu, sto omogucava da se na istom problemu uporede razlicite paradigme masinskog ucenja.

#### Referentni prototipski model

Referentni prototipski model predstavlja oblik klasifikacije zasnovane na sablonima. Za svaku cifru definise se digit prototype, odnosno karakteristicna siva matrica koja predstavlja tipican raspored poteza, a zatim se ulazna cifra poredi sa svim prototipovima. Ovakvi modeli su pogodni za uvodni deo sistema zato sto je njihova logika jednostavna za pracenje: nije neophodno slozeno statisticko treniranje, vec je dovoljno imati reprezentativan sablon i meru slicnosti.

U implementaciji projekta referentni model je dodatno pojednostavljen time sto je baziran na rucno definisanim potezima za svaku cifru. Nakon preprocessing-a ulaza meri se prosecna kvadratna razlika izmedju obradjene slike i svakog prototipa, a zatim se udaljenosti pretvaraju u raspodelu pouzdanosti pomocu softmax funkcije. Prednost ovakvog pristupa je transparentnost: lako je objasniti zasto neka cifra lici na jednu klasu vise nego na drugu. Ogranicenje je to sto uspeh modela zavisi od kvaliteta unapred zadatih sablona i ne moze lako obuhvatiti punu raznovrsnost rukopisa kao nauceni modeli.

#### k-NN klasifikator

k-NN (k-nearest neighbors) klasifikator pripada instancno zasnovanim metodama. On ne gradi eksplicitni skup tezina ili formulu odlucivanja, vec cuva trening primere i za novi ulaz pronalazi one koji su mu najblizi u prostoru osobina. Klasa se zatim odredjuje na osnovu suseda, pa je teorijska osnova ovog modela direktna: slicni primeri treba da pripadaju istoj klasi. [Literatura dopuniti: izvor za k-NN klasifikaciju.]

Bas zato je k-NN dobar kandidat za malo konkretnije implementaciono objasnjenje. U ovom projektu ulaz najpre prolazi kroz preprocessing i PCA redukciju, a zatim se za novi primer racuna euklidsko rastojanje do sacuvanih primera iz trening skupa. Koristi se pet najblizih suseda, pri cemu blizi susedi imaju vecu tezinu od udaljenijih. Prakticna posledica je da model ostaje intuitivan i lako proverljiv: kada pogresi, prirodno je analizirati koji su ga susedni primeri povukli ka pogresnoj odluci. Slabost ovog pristupa je sto predikcija moze postati skuplja kako raste broj trening primera, jer se slicnost mora proceniti nad vise kandidata.

#### Linearni SVM klasifikator

SVM (support vector machine) polazi od ideje da se u prostoru osobina pronadje granica odlucivanja koja sto jasnije razdvaja klase. Kod linearne varijante ta granica je hiperravan, a osnovni cilj je maksimizacija margine, odnosno rastojanja izmedju granice i najkriticnijih primera koji odredjuju njenu poziciju. Ti primeri se nazivaju podrzavajuci vektori i upravo oni nose najvecu informativnu vrednost za formiranje modela.

Za MNIST i slicne visedimenzionalne podatke SVM je teorijski zanimljiv zato sto dobro funkcionise kada je moguce pronaci reprezentaciju u kojoj su klase relativno jasno razdvojive. U ovom projektu zato linearnoj SVM prethodi PCA redukcija na 48 komponenti, cime se originalni pikselski prostor sazimlje pre trazenja razdvajajuce hiperravni. Konkretna implementacija koristi linearnu SVM varijantu sa regularizacionim parametrom 1.0 i ogranicenjem od 5000 iteracija. Takav izbor je razuman kompromis izmedju teorijske snage SVM pristupa i prakticne potrebe da obuka nad MNIST-kompatibilnim podacima ostane pregledna i dovoljno brza.

#### Random Forest klasifikator

Random Forest predstavlja ansamblsku metodu zasnovanu na vecem broju stabala odlucivanja. Svako stablo dobija nesto drugaciji pogled na podatke i kroz niz grananja pokusava da razdvoji klase prema kombinacijama osobina. Konacna odluka ne zavisi od jednog stabla, vec od objedinjavanja njihovih pojedinacnih odluka, zbog cega se smanjuje osetljivost na preprilagodjavanje koje je tipicno za pojedinacna stabla.

U teorijskom smislu prednost ove metode je robusnost: ansambl moze uhvatiti nelinearne odnose medju osobinama bez potrebe da se unapred definise jedna globalna linearna granica odlucivanja. U projektu se koristi suma od 120 stabala sa ogranicenom dubinom od 24 nivoa, a ulaz se pre toga sazimlje PCA projekcijom na 48 komponenti. Time model dobija uravnotezenu kombinaciju dve ideje: najpre se ulazna slika pojednostavljuje u kompaktniji prostor osobina, a zatim vise stabala glasa o klasi na osnovu razlicitih podela tog prostora.

#### MLP neuronska mreza

MLP (multilayer perceptron) pripada dubokim potpuno povezanim neuronskim mrezama. Za razliku od klasifikatora koji odluku zasnivaju na eksplicitnoj meri slicnosti ili na ansamblu stabala, MLP uci niz unutrasnjih transformacija nad ulaznim podacima. Svaki sloj primenjuje linearnu transformaciju, a zatim nelinearnu aktivacionu funkciju, cime se postepeno oblikuju reprezentacije koje bolje razdvajaju klase na izlazu.

U ovom projektu MLP prima 784 ulazne vrednosti, odnosno spljostenu 28x28 sliku, i zatim prolazi kroz dve skrivene potpuno povezane celine dimenzija 128 i 64 neurona sa ReLU aktivacijama. Treniranje se vrsi Adam optimizatorom uz CrossEntropyLoss funkciju greske, u paketima od 32 primera kroz 4 epohe i sa stopom ucenja 0.002. Takva arhitektura nije preterano duboka, ali je dovoljna da pokaze sustinsku ideju neuronskih mreza: model sam uci korisne medjureprezentacije umesto da se u potpunosti oslanja na rucno definisanu metriku slicnosti.

#### CNN neuronska mreza

CNN (convolutional neural network) teorijski je posebno znacajan za klasifikaciju slika zato sto lokalne prostorne odnose obradjuje konvolucionim filtrima umesto da svaki piksel odmah posmatra kao nezavisan ulaz. Konvolucioni slojevi izdvajaju lokalne obrasce, kao sto su ivice, prelazi i jednostavni potezi, dok pooling postupno smanjuje prostornu rezoluciju i zadrzava najvaznije informacije. Na taj nacin mreza stice otpornost na manje translacije i deformacije koje su kod rukopisa uobicajene. [Literatura dopuniti: izvor za CNN i klasifikaciju slika.]

U implementaciji projekta CNN sadrzi dve konvolucione faze. Prva radi nad jednim ulaznim kanalom i proizvodi 8 mapa osobina, a druga iz tih mapa gradi 16 novih mapa, pri cemu se posle svake konvolucije primenjuju ReLU aktivacija i max-pooling. Nakon toga se dobijena reprezentacija spljostava i prosledjuje gustim slojevima dimenzija 32 i 10. Kao i kod MLP modela, obuka se vrsi Adam optimizatorom i CrossEntropyLoss funkcijom, uz 4 epohe i paket od 32 primera, ali sa manjom stopom ucenja od 0.001. U odnosu na MLP, CNN ima teorijsku prednost zato sto arhitektura postuje dvodimenzionalnu prirodu slike i efikasnije koristi lokalnu strukturu rukom pisanih cifara. [Tabela dopuniti: uporedni pregled prototipskog, klasicnih i dubokih modela prema principu rada, prednostima i ogranicenjima.] [Literatura dopuniti: izvori za SVM, Random Forest i MLP.]

### Metrike uspesnosti modela

Evaluacija klasifikatora zahteva vise od jednog broja, jer razlicite metrike osvetljavaju razlicite aspekte ponasanja modela. Najjednostavnija mera je accuracy, odnosno udeo tacno klasifikovanih primera u ukupnom broju evaluiranih primera. Ona daje brz pregled opste uspesnosti, ali ne opisuje podjednako dobro kako se model ponasa po pojedinacnim klasama, posebno kada su greske neravnomerno raspodeljene.

Zbog toga se u ovom projektu, pored accuracy, beleze i macro_precision, macro_recall i macro_f1. Precision opisuje koliki deo primera koje je model svrstao u neku klasu zaista pripada toj klasi, pa je vazan kada je potrebno razumeti koliko su predikcije "ciste". Recall pokazuje koliki deo stvarnih primera neke klase je model uspeo da pronadje, pa govori o osetljivosti modela na tu klasu. F1 mera predstavlja harmonijsku sredinu precision i recall vrednosti i korisna je kada se trazi uravnotezena procena. Prefiks macro oznacava da se vrednosti racunaju zasebno za svaku cifru od 0 do 9, a zatim prosece, cime se svakoj klasi daje isti znacaj u konacnoj oceni modela. Takav izbor je primeren MNIST problemu, jer je cilj da model bude pouzdan nad svim ciframa, a ne samo nad onima koje su lakse za razlikovanje.

Konfuziona matrica predstavlja detaljniji prikaz gresaka klasifikatora. U njoj redovi odgovaraju stvarnim klasama, a kolone predvidjenim klasama, tako da dijagonalni elementi predstavljaju tacne odluke, dok van-dijagonalni elementi otkrivaju koje se cifre medjusobno najcesce mesaju. Upravo zbog toga konfuziona matrica ima interpretativnu vrednost koju zbirne metrike nemaju: ona pokazuje da dva modela mogu imati slicnu ukupnu tacnost, ali razlicit obrazac gresaka. Pored navedenih klasifikacionih mera, backend belezi i prosecno vreme inferencije po primeru, sto je korisno za procenu prakticne upotrebljivosti modela u interaktivnoj aplikaciji. [Literatura dopuniti: izvori za metrike evaluacije i tumacenje konfuzione matrice.]

Na osnovu ovako postavljenog teorijskog okvira moguce je dosledno tumaciti i implementaciju prikazanu u nastavku rada. Razlike izmedju modela vise se ne svode samo na imena algoritama, vec na razlicite nacine reprezentacije ulaza, ucenja granica odlucivanja i procene uspesnosti nad istim MNIST problemom.

## Issue 05 - Tehnologije i razvojno okruzenje

Realizacija projekta zasnovana je na podeli na Python backend i web frontend razvijen u React okruzenju. Ovakav izbor omogucava da se deo zaduzen za obradu podataka, treniranje modela i izlaganje API-ja odvoji od dela koji upravlja korisnickom interakcijom, a da sistem ipak ostane dovoljno jednostavan za lokalno pokretanje na jednoj razvojnoj masini. U repozitorijumu je zato kao glavni backend jezik usvojen Python, dok je frontend izradjen u TypeScript-u uz biblioteku React i razvojno okruzenje Vite.

### Backend tehnologije

Backend koristi FastAPI kao okvir za definisanje HTTP servisa. To se vidi u glavnoj aplikacionoj ulaznoj tacki, gde se formira FastAPI aplikacija i koriste Pydantic modeli za validaciju zahteva kao sto su predikcija sa platna, pregled trening CSV datoteke i parametri treninga. Takav izbor je primeren za projekat ove vrste zato sto omogucava jasan opis ulaznih struktura, strozu validaciju podataka i jednostavno izlaganje REST endpointa potrebnih frontend delu aplikacije.

U fajlu `requirements.txt` eksplicitno su navedene biblioteke FastAPI, Uvicorn, scikit-learn, PyTorch, pytest i httpx. FastAPI i Uvicorn cine serverski sloj: FastAPI definise API, a Uvicorn pokrece ASGI aplikaciju tokom lokalnog razvoja. Biblioteka scikit-learn koristi se za klasicne modele i pomocne tokove obrade podataka, dok se PyTorch koristi za implementacije neuronskih mreza i dubokog ucenja. Pytest je izabran za automatizovanu proveru backend ponasanja, a httpx je ukljucen kao podrska za HTTP test scenarije i komunikaciju sa servisom. Verzije su u repozitorijumu zadate u obliku opsega, cime se ogranicava zavisnost na provereni kompatibilni interval umesto na jednu strogo fiksiranu verziju paketa.

### Frontend tehnologije

Korisnicki interfejs razvijen je u biblioteci React, uz TypeScript kao jezicku osnovu. React omogucava organizovanje interfejsa kao skupa komponenti i pogoduje izgradnji interaktivnih tokova kao sto su crtanje cifre, izbor modela, pregled detalja i upravljanje treningom. TypeScript uvodi tipizaciju na frontend strani, sto smanjuje rizik od gresaka pri radu sa zahtevima i odgovorima backend servisa. Iz fajla `frontend/package.json` vidi se da je razvojno okruzenje zasnovano na Vite alatu, dok su za proveru frontend koda ukljucene biblioteke Vitest, Testing Library i jsdom.

Posebnu ulogu ima i korenski `package.json`, koji ne opisuje samo frontend zavisnosti, vec i celokupnu lokalnu orkestraciju projekta. Komanda `npm run dev` koristi paket `concurrently` kako bi istovremeno pokrenula backend razvojni server i frontend Vite server. Na taj nacin korisnik ne mora rucno da otvara vise terminala niti da odvojeno pokrece dve aplikacije.

### Biblioteke za masinsko ucenje i razvojno okruzenje

Za implementaciju modela koriste se dve glavne biblioteke. Scikit-learn pokriva klasicne pristupe masinskom ucenju i dobro odgovara modelima ciji su artefakti relativno mali i jednostavni za serijalizaciju. PyTorch pokriva duboke modele, odnosno arhitekture koje zahtevaju rad sa tenzorima, optimizatorima i parametrima koji se uce kroz vise epoha. Ovakva podela je u skladu sa ciljem rada: u istoj aplikaciji prikazati i klasicne i duboke pristupe klasifikaciji rukom pisanih cifara, pri cemu backend ostaje jedinstveno mesto za treniranje, evaluaciju i inferenciju.

Razvojno okruzenje prilagodjeno je lokalnom radu na Windows masini. Korenska PowerShell skripta `start.ps1` najpre prelazi u korenski direktorijum projekta, zatim pokusava da pronadje Python 3 preko lokalnog virtuelnog okruzenja, komande `py -3` ili standardne komande `python`, a nakon toga po potrebi kreira `.venv` okruzenje. Sledeci korak je instalacija Python zavisnosti iz fajla `requirements.txt`, zatim provera da li su instalirane Node zavisnosti u korenu projekta i u frontend direktorijumu, i na kraju poziv korenske komande `npm run dev`. Time je ostvarena odluka da `start.ps1` bude jedina vidljiva startup tacka projekta za korisnika na Windows platformi.

Iz skripti se vidi i da se backend tokom razvoja pokrece preko Uvicorn servera na adresi `127.0.0.1:8000`, dok se frontend pokrece preko Vite servera na adresi `127.0.0.1:5173`. Takva podela omogucava jasno razdvajanje prezentacionog i serverskog sloja, ali zadrzava jednostavnost lokalnog monolitnog razvoja, jer se oba dela sistema pokrecu jednom komandom iz istog repozitorijuma.

## Issue 06 - Arhitektura sistema

Arhitektura realizovanog sistema prati obrazac modularnog monolita. Takav opis je opravdan zato sto je resenje organizovano kao jedan repozitorijum i jedna lokalno pokretljiva aplikaciona celina, ali su odgovornosti i tehnicki slojevi ipak jasno razdvojeni. Python backend preuzima serversku logiku, obradu podataka, inferenciju, trening i cuvanje artefakata, dok React frontend upravlja korisnickom interakcijom, prikazom modela i slanjem zahteva prema API-ju. Pored ova dva izvrsna sloja postoji i trajni lokalni storage sloj u direktorijumu `data`, gde se nalaze shipped modeli, custom modeli i centralni registar modela. Time je postignut kompromis izmedju jednostavnog lokalnog pokretanja i dovoljno pregledne unutrasnje strukture sistema.

### Arhitektonski stil i raspodela odgovornosti

Korenska skripta `start.ps1` predstavlja jedinu vidljivu startup tacku sistema. Ona najpre priprema Python virtuelno okruzenje i Python zavisnosti, zatim proverava prisustvo Node paketa u korenu projekta i u frontend direktorijumu, i na kraju pokrece korensku razvojnu komandu `npm run dev`. Na taj nacin korisnik ne upravlja odvojeno backend i frontend procesima, vec sistem posmatra kao jednu celinu koja se podize iz korena repozitorijuma. Iza takvog ulaza ipak ostaje jasna podela slojeva: backend radi kao FastAPI servis, frontend kao Vite/React aplikacija, a trajni podaci o modelima cuvaju se u projektno-lokalnim JSON i binarnim artefaktima umesto u spoljnoj bazi podataka.

U fajlu `backend/app/main.py` definisani su endpointi za proveru zdravlja sistema, ucitavanje modela, predikciju, pregled trening CSV datoteke, pokretanje trening posla, pracenje statusa treninga i brisanje custom modela. Frontend zbog toga ne sadrzi poslovnu logiku klasifikacije, vec je odgovoran za unos podataka, izbor modela, validaciju osnovnih uslova interakcije i prikaz rezultata. Ovakva podela smanjuje dupliranje pravila obrade i omogucava da jedna backend implementacija bude zajednicka za sve korisnicke tokove.

### Backend servis i storage sloj

Prilikom podizanja aplikacije backend najpre obezbedjuje minimalnu storage strukturu. Funkcija `ensure_storage_structure` kreira direktorijume `shipped-models`, `custom-models` i `registry`, a zatim proverava da li je referentni shipped artefakt dostupan. U istom lifecycle koraku formira se i instanca `TrainingJobManager`, sto znaci da backend od samog pocetka preuzima i upravljanje asinhronim treniranjem. Endpoint `/api/health` zatim vraca ne samo informaciju da je servis dostupan, nego i da li je storage spreman, koja je njegova lokacija i koji su obavezni poddirektorijumi prisutni. Test `backend/tests/test_health.py` potvrduje upravo taj ugovor izmedju startup faze i radnog storage okruzenja.

Posebnu ulogu u arhitekturi ima razdvajanje javnih metapodataka od izvrsnih artefakata modela. Shipped modeli cuvaju se pod `data/shipped-models`, pri cemu JSON metapodaci sadrze putanju do runtime artefakta, dok se sami binarni fajlovi koriste tek kada backend treba da izvrsi inferenciju. Custom modeli upisuju se u `data/custom-models`, zajedno sa odgovarajucim JSON opisom i, po potrebi, dodatnim serijalizovanim artefaktom kao sto je `pkl.gz` ili PyTorch checkpoint. Takvo resenje je arhitektonski vazno zato sto korisnicki interfejs radi nad jedinstvenim opisom modela, dok backend interno zadrzava detalje neophodne za izvrsavanje i ucitavanje modela.

### Model katalog i model registry

Centralni objekat koji povezuje storage sloj i korisnicke tokove jeste model registry. Fajl `data/registry/models.json` ne cuva kompletne modele, vec mapira identitet modela, njegovu vrstu i putanju do pripadajuceg JSON opisa artefakta. U skladu sa terminologijom iz `CONTEXT.md`, registry predstavlja lokalni zapis o tome gde se svaki model iz model kataloga nalazi, dok model katalog predstavlja javnu listu shipped i custom modela koju frontend prikazuje korisniku. Funkcija `ensure_model_registry` u `backend/app/custom_training.py` pri svakom radu sa katalogom spaja shipped unose i postojece registry stavke, uklanja duplikate i upisuje samo validne putanje. Zbog toga built-in i custom modeli prolaze kroz isti javni tok prikaza, izbora i detalja, iako se fizicki nalaze u razlicitim poddirektorijumima.

Takva organizacija omogucava da katalog modela bude zajednicka tacka sistema. Endpoint `/api/models` poziva `list_available_models`, koji iz registra cita sve poznate modele, proverava da li su njihovi metapodaci i runtime artefakti zaista prisutni i zatim vraca samo javni deo metapodataka. Posledica ove odluke je da leaderboard, detalji modela i padajuci izbor modela na frontend strani ne moraju da znaju da li je konkretan model shipped ili custom, niti kojom je bibliotekom implementiran. Arhitektura je zato usmerena na zajednicki ugovor nad modelima, a ne na posebne ekrane i posebne API-je za svaku porodicu klasifikatora. [Slika dopuniti: arhitektonski dijagram sa frontend slojem, FastAPI backend slojem i lokalnim storage slojem `data/shipped-models`, `data/custom-models` i `data/registry`.]

### Frontend kao orkestrator korisnickih tokova

Frontend aplikacija iz fajla `frontend/src/App.tsx` pri ucitavanju paralelno poziva `/api/health` i `/api/models`, pa na osnovu tih odgovora formira pocetno stanje interfejsa i podrazumevani izbor modela. Time se odmah uspostavlja veza izmedju prezentacionog sloja i zajednickog model kataloga. U nastavku rada frontend ostaje relativno tanak sloj: korisnik moze da crta cifru na platnu, da izabere model iz zajednicke liste, da pregleda leaderboard i detalje, ili da predje u rezim treninga gde uploaduje CSV, proverava podelu skupa i pokrece novi trening. Sve kljucne odluke o validnosti ulaza, inferenciji i perzistenciji ostaju na backend strani.

### Tok korisnickog zahteva kroz sistem

Tok predikcije pokazuje kako saradjuju svi slojevi sistema. Korisnik najpre nacrta cifru na 20x20 platnu i izabere model iz zajednickog kataloga. Frontend zatim formira zahtev prema `/api/predict`, u koji ukljucuje identifikator modela i matricu piksela. Backend na osnovu tog identifikatora bira odgovarajuci put: referentni prototipski model, shipped klasicki model, shipped deep model ili custom model. Nakon zajednickog preprocessing-a backend vraca strukturisan odgovor koji sadrzi javne informacije o modelu, predvidjenu cifru i raspodelu pouzdanosti po klasama, a frontend te rezultate prikazuje bez potrebe da poznaje unutrasnju implementaciju modela.

Tok treninga je slozeniji i jos jasnije pokazuje arhitektonske granice. U trening rezimu frontend najpre salje CSV sadrzaj na `/api/training/csv-preview`, kako bi backend proverio format i izracunao podelu na train, validation i test skup. Tek nakon toga frontend moze poslati zahtev prema `/api/training/jobs`, zajedno sa nazivom modela, izabranom porodicom klasifikatora, semenom i kuriranim hiperparametrima. `TrainingJobManager` takav zahtev pretvara u asinhroni posao, vodi racuna da u verziji jedan postoji samo jedan aktivan trening i po zavrsetku trajno upisuje metapodatke i runtime artefakt novog modela. Kada posao predje u stanje `completed`, frontend periodicnim polling-om osvezava bootstrap podatke i time automatski dobija novi model u istom katalogu i na istom leaderboard-u. Ovakav tok pokazuje da je ceo sistem projektovan oko jedinstvenog model lifecycle-a: model se pojavljuje, evaluira, bira i eventualno brise kroz zajednicku arhitekturu, bez obzira na to da li pripada shipped ili custom grupi.

Na osnovu ovakve organizacije moze se zakljuciti da arhitektura sistema nije slucajno nastala kao skup odvojenih skripti, vec kao dosledno povezan lokalni softverski sistem. Startup ulaz, backend API, frontend tokovi i storage sloj grade jednu funkcionalnu celinu u kojoj su pravila zajednicka, a implementacione razlike izmedju porodica modela zadrzane iza jasnih apstrakcionih granica. To predstavlja odgovarajucu osnovu za detaljniji prikaz backend implementacije i konkretnih API tokova u narednom poglavlju.

## Issue 07 - Backend implementacija

Backend predstavlja centralno implementaciono jezgro sistema, jer objedinjuje validaciju ulaza, pripremu podataka, inferenciju, trening, evaluaciju i upis modelskih artefakata. Za razliku od frontend dela, koji pre svega upravlja korisnickim tokovima i prikazom rezultata, backend sadrzi pravila po kojima se svaki model pojavljuje u katalogu, bira za predikciju, trenira, cuva i po potrebi uklanja. Zbog toga je u tehnickom smislu upravo backend mesto na kome se spajaju produktne odluke iz `PRD.md`, terminologija iz `CONTEXT.md` i razvojni inkrementi zabelezeni kroz zavrsene issue-e.

### API sloj i validacija zahteva

Ulazna tacka backend servisa nalazi se u fajlu `backend/app/main.py`, gde se formira FastAPI aplikacija i definise njen lifecycle. U istom modulu nalaze se Pydantic modeli koji backend ugovor cine eksplicitnim. `CanvasPayload` proverava dimenzije platna i broj piksela, `TrainingSplitPayload` namece pravilo da zbir train, validation i test odnosa mora biti jednak 1.0, a `CustomTrainingRequest` dodatno vezuje izbor hiperparametara za izabranu porodicu modela. Time validacija ne ostaje na nivou implicitnih pretpostavki unutar pojedinacnih funkcija, vec se podize na nivo samog API interfejsa.

Iz perspektive ruta, backend je organizovan oko nekoliko jasno odvojenih tokova. Ruta `/api/health` proverava spremnost storage strukture i predstavlja prvi ugovor koji frontend koristi pri bootstrap fazi. Ruta `/api/models` vraca zajednicki katalog raspolozivih modela. Ruta `/api/predict` prihvata identifikator modela i platno sa pikselima, dok `/api/training/csv-preview` sluzi za proveru i sumarizaciju trening CSV datoteke pre same obuke. Za dugotrajnije operacije uveden je odvojen trening tok: `/api/training/jobs` pokrece asinhroni trening posao, `/api/training/jobs/{job_id}` vraca njegovo trenutno stanje, a `DELETE /api/models/{model_id}` omogucava uklanjanje samo custom modela. Na taj nacin su citanje stanja sistema, inferencija i modifikacija modelskog kataloga razdvojeni u zasebne, ali medjusobno povezane API celine.

Vazan implementacioni detalj je i nacin obrade gresaka. `main.py` dosledno prevodi lokalne backend izuzetke u HTTP odgovore koji su znacajni za korisnicki interfejs: nepoznat model vraca 404, neispravan unos 400, dok sukob naziva modela ili pokusaj paralelnog treninga vraca 409. Takav pristup omogucava da frontend prikazuje smislen povratni kontekst bez potrebe da poznaje interne klase izuzetaka. [Prilog dopuniti: primeri JSON zahteva i odgovora za `/api/predict`, `/api/training/csv-preview` i `/api/training/jobs`.]

### Referentni preprocessing i prvi predikcioni slice

Prvi potpuni backend slice projekta vidi se u modulu `backend/app/reference_model.py`, koji implementira referentni prototipski klasifikator. Njegov znacaj nije samo u tome sto daje bazni shipped model, vec i u tome sto uspostavlja kanonski preprocessing za ulaz sa platna. Funkcija `preprocess_canvas` najpre normalizuje intenzitete, zatim pronalazi ogranicavajuci okvir aktivnih piksela, izdvaja sadrzaj, skalira ga u ciljnu sliku 28x28 sa unutrasnjim okvirom 20x20 i translira ga prema centru mase. Time se rucno crtani ulaz svodi na reprezentaciju koja je uporediva sa MNIST zapisima i pogodna za sve naredne modele.

Nad tako pripremljenim ulazom referentni model koristi rucno definisane potezne prototipove za svaku cifru. Funkcija `predict_reference_digit` racuna raspodelu pouzdanosti i vraca najverovatniju klasu zajedno sa javnim metapodacima modela. Time je jos u okviru development issue-a broj 2 uspostavljen potpuni backend ugovor za inferenciju: korisnik salje sirove piksele platna, backend obavlja standardizaciju ulaza, primenjuje model i vraca strukturisan odgovor sa predikcijom i poverenjem po klasama. Taj obrazac kasnije postaje zajednicka osnova i za ostale porodice modela.

### CSV intake i preview podele skupa

Poseban backend podsistem zaduzen je za rad sa trening skupovima u CSV formatu. Modul `backend/app/training_csv.py` precizira MNIST-kompatibilnu semu kao jedan `label` stubac i tacno 784 piksel-vrednosti `pixel0` do `pixel783`. Funkcija `parse_training_csv_rows` redom proverava zaglavlje, broj kolona, opseg oznake klase od 0 do 9 i dozvoljeni opseg intenziteta od 0 do 255. Time backend odbacuje nekompatibilne datoteke pre nego sto takvi podaci dodju do trening logike.

Na toj osnovi funkcija `preview_training_csv` gradi odgovor za rutu `/api/training/csv-preview`. Povratni payload sadrzi naziv datoteke, broj primera, broj osobina, raspon oznaka klasa i broj elemenata po train, validation i test delu. U razvojnom smislu ovo odgovara ciljevima issue-a broj 4, gde je trening CSV intake uveden kao poseban korak pre same obuke. Testovi iz `backend/tests/test_training_csv_preview.py` dodatno potvrduju i pozitivni scenario pregleda podele i negativne scenarije neispravne sheme ili nevalidnih oznaka klasa, sto znaci da CSV preview nije samo pomocna UI funkcija, vec eksplicitno verifikovan backend ugovor.

### Zajednicki model lifecycle i adapteri inferencije

Jedna od kljucnih osobina backend implementacije jeste to sto svi modeli prolaze kroz zajednicki lifecycle bez obzira na porodicu algoritma. Tu ulogu nose `backend/app/custom_training.py` i `backend/app/shipped_classical_models.py`. Funkcija `ensure_model_registry` odrzava centralni registar modela, `list_available_models` vraca javni katalog, dok `predict_available_model` radi centralni dispatch za inferenciju. Ako je zatrazena referentna prototipska varijanta, poziva se `predict_reference_digit`; ako je u pitanju shipped klasicki ili deep model, aktiviraju se odgovarajuci adapteri; a ako je izabran custom model, porodica modela se odredjuje iz njegovih metapodataka pa se primenjuje isti preprocessing i odgovarajuci klasifikacioni put.

Takav pristup je posebno vazan zato sto uklanja potrebu za posebnim endpointima po tipu modela. Backend prema spolja izlaze jedinstvenu rutu za predikciju i jedinstvenu rutu za listu modela, dok unutar sebe skriva razlike izmedju prototipskih, klasicnih i dubokih implementacija. Posledica je da katalog modela, leaderboard i detalji modela na frontend strani rade nad jednim ugovorom, dok se arhitektonske razlike resavaju iza adapter sloja. Upravo to je trazeno i u development issue-ima broj 6 i 7, gde se shipped klasicki i deep modeli uvode u postojeci tok bez posebnih ekrana i bez posebne spoljne logike.

### Shipped klasicki i shipped deep modeli

Shipped klasicki modeli podrzani su kroz modul `backend/app/shipped_classical_models.py`, koji validira njihove JSON metapodatke, proverava prisustvo runtime artefakta i po potrebi materijalizuje kanonske shipped modele u aktivni storage root. Isti modul sadrzi i izvrsne adaptere za klasicku i duboku inferenciju: za sklearn modele ucitava se serijalizovani estimator i iz njega izvodi raspodela pouzdanosti, dok se za duboke modele ucitava PyTorch checkpoint i rekonstruisu MLP ili CNN arhitekture pre racunanja logita i softmax raspodele.

Sama izgradnja shipped modela organizovana je kroz odvojene trening module. `backend/app/shipped_classical_training.py` regenerise k-NN, linearnu SVM i Random Forest modele iz repozitorijumskog `train.csv`, uz unapred definisane PCA konfiguracije, metrike, konfuzionu matricu, uzorke predikcija i putanju do `pickle-gzip` artefakta. `backend/app/shipped_deep_training.py` na slican nacin regenerise MLP i CNN modele, ali sa PyTorch checkpoint artefaktima i dodatnim deep metapodacima kao sto su `architecture_summary` i `epoch_curves`. Time backend zadrzava zajednicki javni format modelskih opisa, ali istovremeno dozvoljava ogranicene deep-specificne dodatke koji su potvrdjeni i kroz issue broj 7.

### Asinhroni custom training i persistencija

Najopsezniji backend tok realizovan je u `backend/app/custom_training.py`, gde je smeštena orkestracija custom treninga. Klasa `TrainingJobManager` uvodi pravilo da u verziji jedan moze postojati samo jedan aktivan trening posao. Prilikom pokretanja novog posla backend proverava da li je naziv modela jedinstven, generise interni `model_id`, formira snapshot posla i obuku prebacuje u zasebnu pozadinsku nit. Tok izvrsavanja podeljen je na faze validacije skupa, mesanja i podele podataka, same obuke i upisa metapodataka, a svaka od tih faza ostavlja procenat napretka i tekstualni opis trenutnog stanja.

Po zavrsetku parsiranja i podele podataka backend gradi model u zavisnosti od porodice klasifikatora. Prototipski custom modeli ostaju bez izdvojenog runtime fajla, jer se njihovi prototipovi cuvaju direktno u metapodacima. Klasicni custom modeli cuvaju se kao `pkl.gz` artefakti, dok duboki custom modeli koriste zasebne checkpoint fajlove. U sva tri slucaja backend pored runtime artefakta upisuje i bogat JSON opis sa izvorom skupa, brojem primera po split-u, kuriranim hiperparametrima, evaluacionim metrikama, konfuzionom matricom, uzorcima predikcija i snimkom trening konfiguracije. Funkcija `persist_custom_artifact` zatim upisuje taj opis u `data/custom-models` i azurira model registry, cime novi model odmah postaje deo zajednickog kataloga.

Backend u istom modulu implementira i kontrolisano brisanje modela. Funkcija `delete_custom_model` eksplicitno zabranjuje uklanjanje built-in modela, uklanja JSON opis custom modela, po potrebi brise i njegov runtime artefakt i zatim osvezava registry zapis. Time je lifecycle custom modela zatvoren unutar jednog modula: od kreiranja zahteva, preko asinhrone obuke i upisa, do kasnijeg uklanjanja. Testovi iz `backend/tests/test_custom_training.py` potvrduju upravo ovu celinu, ukljucujuci nastanak modela, njegovo pojavljivanje u zajednickom katalogu, mogucnost predikcije nad novim modelom i potpuno uklanjanje metapodataka i runtime fajlova nakon brisanja.

### Backend verifikacija

Automatizovana verifikacija backend-a prati istu modularnu podelu kao i implementacija. Test `backend/tests/test_health.py` potvrduje inicijalizaciju storage strukture i odgovor health rute. `backend/tests/test_training_csv_preview.py` pokriva validaciju trening CSV ulaza i preview podele skupa. `backend/tests/test_custom_training.py` proverava kompletan asinhroni trening tok, deljenje kataloga modela i brisanje custom artefakata. Dodatno, testovi za shipped klasicke i shipped deep trening module proveravaju da se pri regeneraciji zaista upisuju odgovarajuci metapodaci i runtime fajlovi. Takva raspodela testova pokazuje da backend nije proveren samo na nivou pojedinacnih pomocnih funkcija, vec i na nivou ugovora koje frontend i ostatak sistema direktno koriste.

Na osnovu ovako organizovanih modula moze se zakljuciti da backend implementacija nije skup nepovezanih funkcija za rad sa modelima, vec centralni aplikacioni sloj koji drzi pravila klasifikacije, validacije i upravljanja modelima na jednom mestu. Upravo zbog toga naredno poglavlje o frontend pregledu moze ostati krace: korisnicki interfejs prikazuje stanje i pokrece tokove, ali glavna tehnicka slozenost sistema ostaje koncentrisana u backend-u.

## Issue 08 - Frontend pregled

Frontend deo sistema ima ulogu prezentacionog i orkestracionog sloja koji objedinjeno prikazuje stanje backend-a, modela i aktivnih korisnickih tokova, ali pri tome ne preuzima samu klasifikacionu logiku. U glavnoj React komponenti aplikacije pri inicijalnom ucitavanju paralelno se pozivaju rute `/api/health` i `/api/models`, nakon cega interfejs formira pocetno stanje aplikacije, prikazuje indikator spremnosti sistema i bira podrazumevani model iz zajednickog Model Catalog-a. Na taj nacin korisnik odmah dobija informaciju da li je servis dostupan i koji su Shipped Models i Custom Models trenutno spremni za pregled i rad.

### Organizacija interfejsa

Vizuelna organizacija interfejsa zasniva se na jasno odvojenim panelima. Gornji hero deo saopstava osnovnu namenu aplikacije i trenutno stanje sistema, dok se ispod njega radna povrsina deli na levi deo za aktivni tok rada i desni deo za izbor modela, detalje i rezultat predikcije. Stilovi iz `frontend/src/App.css` realizuju dvokolonski raspored za sire ekrane, a pri manjim sirinama ga automatski prevode u jednokolonski prikaz, cime interfejs ostaje upotrebljiv i na uzim ekranima. Vizuelni jezik zasnovan je na karticama, statusnim oznakama i tabelarnim prikazima, sto odgovara demonstracionom karakteru projekta: naglasak nije na dekorativnim elementima, vec na citljivom prikazu metapodataka i stanja rada. [Slika dopuniti: pocetni prikaz aplikacije sa hero panelom, levim radnim prostorom, leaderboard-om i karticom detalja modela.]

Pored samog rasporeda vazna je i odluka da prediction i training ne budu prikazani kao dve odvojene stranice, vec kao dva rezima rada unutar iste radne kartice. Time se korisniku jasno stavlja do znanja da je rec o istom sistemu i istom modelskom katalogu, dok se istovremeno izbegava preopterecenje interfejsa stalnim prikazom svih kontrola odjednom. U predikcionom rezimu istaknute su kontrole za crtanje cifre i pokretanje jedne Prediction Run operacije, dok trening rezim tu istu povrsinu koristi za upload Training Dataset-a, preview Dataset Split-a i unos kuriranih hiperparametara. Ovakva podela je u skladu sa razvojnim sledom issue-a broj 4 i broj 5, gde je trening tok uveden kao prosirenje postojeceg levog radnog prostora, a ne kao poseban ekran.

### Rezim predikcije i izbor modela

U rezimu predikcije korisnik radi nad platnom dimenzija 20x20, koje je na frontend strani predstavljeno kao mreza interaktivnih polja. Interfejs omogucava pojedinacno bojenje ili povlacenje preko vise polja, kao i brisanje cele povrsine pre nove probe. Kada korisnik izabere model i pokrene predikciju, frontend salje sirovu matricu piksela zajedno sa identifikatorom izabranog modela prema ruti `/api/predict`, dok backend zatim obavlja standardizaciju, centriranje i samo bodovanje ulaza. Time je odrzana vazna arhitektonska odluka da canonical preprocessing ostane na serverskoj strani, a da frontend ostane zaduzen za unos i prikaz rezultata.

Izbor modela u interfejsu nije odvojen od ostatka sistema, vec koristi isti skup unosa koji je prisutan i u leaderboard-u. Padajuca lista modela prikazuje naziv i vrstu svakog unosa, a izbor modela istovremeno odredjuje koji ce se metapodaci prikazati u desnoj kartici i koji ce model biti iskoriscen za narednu Prediction Run operaciju. Nakon uspesnog odgovora backend-a frontend prikazuje predvidjenu cifru i kompletnu raspodelu pouzdanosti za cifre od 0 do 9, sto korisniku omogucava da ne vidi samo konacnu odluku vec i obrazac nesigurnosti modela.

### Trening rezim i upravljanje custom modelima

Prelaskom u training mode leva radna povrsina vise ne prikazuje canvas, vec formu za rad sa MNIST-kompatibilnim CSV skupom. Korisnik najpre bira datoteku, zatim po potrebi menja podrazumevane split odnose 80/10/10 i pokrece preview preko rute `/api/training/csv-preview`. Frontend na osnovu odgovora backend-a prikazuje broj primera, broj osobina, opseg oznaka klasa i tacne velicine train, validation i test dela. Ukoliko CSV ne odgovara propisanom formatu, greska se ne skriva u pozadini, vec se prikazuje kao neposredna povratna informacija u istoj kartici. [Slika dopuniti: trening rezim sa CSV upload-om, preview karticama i statusom aktivnog Training Job-a.]

Tek nakon uspesnog preview koraka korisnik moze pokrenuti Training Job. Interfejs podrzava izbor vise porodica modela, ukljucujuci reference prototype, k-NN, SVM, Random Forest, MLP i CNN, pri cemu svaka porodica otvara samo njoj relevantan skup kuriranih hiperparametara. Vazno version-one pravilo jeste da se naziv Custom Model-a uvek unosi rucno; aplikacija ne generise podrazumevana imena i ne sakriva identitet eksperimenta iza automatskih oznaka. Pored toga, frontend prikazuje meka upozorenja za potencijalno rizicne vrednosti, na primer za prevelik broj PCA komponenti, previsok broj stabala ili preduge deep treninge, cime korisnik dobija smernice bez uvodjenja nepotrebno restriktivnih zabrana.

Kada trening zapocne, frontend salje zahtev na `/api/training/jobs`, a zatim status asinhrono osvezava periodickim polling-om rute `/api/training/jobs/{job_id}`. Time se u interfejsu pojavljuju tekstualna faza posla i progress bar, dok ostatak aplikacije ostaje responzivan. Uspesan zavrsetak automatski povlaci osvezavanje bootstrap podataka i novi Custom Model se odmah pojavljuje u istom picker-u i istom leaderboard-u kao i ostali modeli. Ako trening ne uspe, greska ostaje prikazana samo kao inline povratna informacija za tekuci pokusaj i ne stvara lazni ili neuspesni unos u Model Catalog-u, sto je uskladjeno sa verzijom jedan definisanom u `docs/README.md` i issue-u broj 10.

### Leaderboard, detalji i frontend verifikacija

Jedna od najvaznijih frontend odluka jeste da leaderboard ostane stalno vidljiv i tokom trening rezima. U okviru iste stranice korisnik moze da sortira raspolozive modele po accuracy, macro F1 ili prosecnom vremenu inferencije, a izbor reda u tabeli automatski osvezava desnu karticu sa detaljima modela. Tako leaderboard ne predstavlja samo preglednu tabelu, vec centralnu ulaznu tacku za uporedjivanje modela i prelazak ka dubljem tumacenju njihovih osobina.

Kartica detalja prikazuje izvor skupa, vreme treniranja, velicinu train, validation i test delova, ulazni oblik, metrike, hiperparametre, konfuzionu matricu i uzorke predikcija. Kod custom modela dodatno se prikazuje training snapshot sa sacuvanim seed-om, nazivom otpremljene datoteke i konfiguracijom podele skupa, dok se za deep modele aktivira tabbed prikaz sa dve sekcije: opstim pregledom i deep detaljima. U tom drugom prikazu nalaze se architecture summary i epoch curves, cime je isti obrazac detalja iskoriscen i za shipped i za custom duboke modele. Ako je selektovan Custom Model, korisnik iz iste kartice moze da pokrene i njegovo brisanje, dok su Shipped Models od takve akcije zasticeni.

Frontend verifikacija sprovedena je kroz testove u `frontend/src/App.test.tsx`, koji pokrivaju bootstrap pozive prema `/api/health` i `/api/models`, tok slanja nacrtane cifre na predikciju, sortiranje leaderboard-a i prikaz detalja modela, otvaranje deep tabs prikaza, zamenu canvas rezima trening rezimom, CSV validaciju i split preview, rucno imenovanje custom modela, inline prikaz neuspelog treninga bez stvaranja neuspesnog unosa i kompletan zajednicki tok nastanka i brisanja custom modela. Zbog toga frontend nije verifikovan samo na nivou statickog prikaza, vec i kroz glavne korisnicke scenarije koji direktno povezuju interfejs sa backend ugovorima.

## Issue 09 - Faze implementacije

Razvoj projekta nije tekao kao jednokratna izrada kompletnog sistema, vec kao niz zavrsenih razvojnih issue-a od kojih je svaki donosio mali, ali funkcionalan inkrement. Takav pristup je bio u skladu sa planerskim odlukama iz `PRD.md` i terminologijom iz `CONTEXT.md`: jos na nivou planiranja bili su definisani zajednicki Model Catalog, backend kao mesto kanonskog preprocessing-a, lokalni storage za artefakte u okviru projekta i ogranicenje da u verziji jedan postoji samo jedan aktivan Training Job. Razvojni issue-i zato nisu predstavljali spisak nepovezanih zadataka, vec operativnu razradu istog plana kroz proverljive tehnicke faze.

[Tabela dopuniti: sazet pregled deset razvojnih faza, glavnog problema koji svaka faza resava, uvedenog resenja i neposrednog rezultata.]

### Faza 1 - Pokretljiv aplikacioni shell

Prva faza resavala je osnovni problem pokretljivosti i integracije. Pre nego sto je bilo moguce prikazati bilo kakvu klasifikaciju, bilo je neophodno obezbediti da se aplikacija pokrece iz korena repozitorijuma na Windows masini i da frontend i backend uspostave vidljivu vezu. Zbog toga je uveden root startup tok preko `start.ps1`, minimalni frontend shell i health ugovor backend-a. Neposredan rezultat ove faze bio je funkcionalan aplikacioni okvir u kome je moglo da se proveri podizanje servisa, inicijalizacija storage strukture i osnovna komunikacija izmedju dva sloja sistema.

### Faza 2 - Referentni shipped model i prvi tok predikcije

Druga faza uvodi prvi potpuni Prediction Run tok. Problem vise nije bio samo da aplikacija radi, vec da korisnik moze da nacrta cifru, posalje Canvas Digit backend-u i dobije smislen rezultat klasifikacije. Kao resenje uveden je referentni prototipski Shipped Model zajedno sa kanonskim preprocessing-om ulaza i odgovorom koji sadrzi predvidjenu cifru i raspodelu pouzdanosti za klase od 0 do 9. Rezultat ove faze bio je uspostavljen osnovni inferencioni ugovor na kome su kasnije gradjene sve ostale porodice modela.

### Faza 3 - Leaderboard i detalji shipped modela

Treca faza prosiruje sistem sa pojedinacne demonstracije predikcije na uporedni prikaz modela. Pokazalo se da jedna izdvojena predikcija nije dovoljna ako korisnik treba da razume razlike izmedju modela i da tumaci njihove osobine. Resenje je bilo uvodjenje leaderboard prikaza sa sortiranim metrikama i kartice detalja modela koja prikazuje izvor skupa, vreme treniranja, hiperparametre, konfuzionu matricu i uzorke predikcija. Time je Model Catalog dobio svoju prvu punu korisnicku formu, a backend metapodaci su postali direktno pregledni u interfejsu bez posebnih ekrana po modelu.

### Faza 4 - CSV intake i preview podele skupa

Cetvrta faza priprema ulaz u trening tok tako sto uvodi rad sa Training Dataset upload-om. Problem koji je trebalo resiti bio je kako korisniku omoguciti eksperimentisanje nad sopstvenim MNIST-kompatibilnim CSV skupom, a da se pri tome ne dopusti da neispravan format stigne do same obuke. Kao resenje uvedeni su validacija CSV seme, prikaz gresaka za neispravne oznake ili kolone i preview Dataset Split-a sa podesivim train, validation i test odnosima. Rezultat ove faze bio je kontrolisani ulaz u trening, pri cemu training mode u interfejsu zamenjuje prediction canvas i korisniku daje jasan pregled podataka pre pokretanja obuke.

### Faza 5 - Prvi custom training slice

Peta faza predstavlja prelaz sa statickog prikaza shipped modela na dinamicni model lifecycle. Osnovni problem sada je bio kako od validiranog Training Dataset-a doci do novog Custom Model-a koji ce imati sopstveni identitet, sacuvan seed, config snapshot i vidljivo stanje napretka. Resenje je bilo uvodjenje asinhronog Training Job toka za prvi referentni klasifikator, sa pravilom da u verziji jedan moze postojati samo jedan aktivan posao, uz upis novog modela u zajednicki katalog i mogucnost kasnijeg brisanja. Rezultat ove faze bio je da sistem vise nije samo demonstrator gotovih modela, vec alat u kome korisnik moze da pokrene sopstvenu obuku i odmah vidi novi unos u istom picker-u i na istom leaderboard-u.

### Faza 6 - Prosirenje shipped klasicnih modela

Sesta faza sirila je opseg vec postojece infrastrukture na dodatne klasicne algoritme. Nakon sto su picker, leaderboard i detalji vec postojali, sledeci problem bio je da se katalog ne zadrzi samo na referentnom prototipu. Resenje je bilo uvodjenje k-NN, linearne SVM i Random Forest varijante kroz isti Model Registry, isti tok predikcije i isti detaljni prikaz bez posebnih UI grana. Rezultat ove faze bio je da klasicni modeli postanu ravnopravni clanovi zajednickog kataloga, sto je omogucilo smislenije poredjenje razlicitih pristupa na istom problemu.

### Faza 7 - Prosirenje shipped deep modela

Sedma faza nadovezuje se na prethodnu, ali uvodi slozeniju grupu modela. Problem je bio kako uklopiti MLP i CNN u vec postojeci zajednicki ugovor, a da se pri tome zadrze samo one deep-specificne informacije koje zaista imaju interpretativnu vrednost. Kao resenje uvedeni su shipped deep modeli sa istim tokovima izbora, predikcije i leaderboard prikaza, dok su dodatne informacije kao sto su architecture summary i epoch curves smestene u tabbed prikaz kartice detalja. Rezultat ove faze bio je da duboki modeli budu ukljuceni u sistem bez narusavanja koherentnosti interfejsa ili backend adapter sloja.

### Faza 8 - Prosirenje custom training-a na klasicne modele

Osma faza prenosi vec uspostavljeni custom training obrazac sa referentnog klasifikatora na klasicne modele. Problem vise nije bio u samoj orkestraciji asinhrone obuke, vec u tome da razlicite klasicne porodice modela koriste isti lifecycle, iste granice verzije jedan i isti nacin pojavljivanja u zajednickom katalogu. Resenje je bilo prosirenje trening toka na k-NN, SVM i Random Forest uz kurirane hiperparametre, perzistenciju odgovarajucih `pkl.gz` artefakata i zadrzavanje zajednickih pravila za imenovanje, registry i brisanje. Rezultat ove faze bio je da korisnik moze da kreira i uporedjuje vise custom klasicnih eksperimenata bez napustanja istog radnog toka.

### Faza 9 - Prosirenje custom training-a na duboke modele

Deveta faza zaokruzuje implementacioni opseg custom treninga uvodjenjem MLP i CNN varijanti u isti asinhroni lifecycle. Razvojni issue broj 9 trazio je da duboki custom modeli ne dobiju izdvojen tretman, vec da koriste isti zajednicki izbor modela, isti leaderboard i isti obrazac detalja koji je ranije uspostavljen za shipped deep modele. Resenje je realizovano tako sto backend custom training tok prepoznaje duboke porodice modela, trenira ih nad dostavljenim skupom, cuva PyTorch checkpoint artefakte i u javnim metapodacima izdvaja `deep_details` sa `architecture_summary` i `epoch_curves`, dok frontend te dodatke prikazuje kroz postojeci tabbed prikaz u kartici detalja. Rezultat ove faze bio je da custom deep modeli postanu potpuno uporedivi sa shipped deep modelima i da projektovani zajednicki Model Catalog zaista obuhvati i klasicne i duboke eksperimente korisnika.

### Faza 10 - Zavrsno ciscenje i potvrda version-one pravila

Deseta faza nije donosila novu porodicu modela, vec je stabilizovala proizvodni opseg verzije jedan. Problem u ovoj tacki nije bio nedostatak funkcionalnosti, vec potreba da se nekoliko otvorenih produktnih odluka definitivno zatvori i dosledno sprovede kroz dokumentaciju, backend i interfejs. Kao resenje potvrdena su pravila da `start.ps1` ostaje jedina vidljiva startup tacka, da Custom Model mora imati rucno uneto ime i da neuspesni trening ostaje samo inline povratna informacija bez trajnog neuspesnog unosa u Model Catalog-u. Rezultat ove faze bio je ciscenje ivicnih ponasanja i jasnije definisana granica verzije jedan, sto je projekat ucinilo konzistentnijim i pogodnijim za demonstraciju.

Posmatrane zajedno, ove faze pokazuju da je implementacija pratila disciplinovan inkrementalni razvoj: svaka sledeca etapa oslanjala se na prethodno potvrdjen ugovor umesto da uvodi novi paralelni tok. Na taj nacin su planerske odluke iz `PRD.md`, kao sto su zajednicki Model Catalog, backend-owned preprocessing, kurirani hiperparametri i ogranicen scope verzije jedan, postepeno prevedene u konkretan kod, testove i korisnicke tokove koji su vidljivi u repozitorijumu. Zbog toga pregled faza implementacije nije samo hronoloski rezime, vec i pokazatelj da je projekat razvijan kao povezana softverska celina sa jasnim tehnickim prioritetima.

## Issue 10 - Testiranje i verifikacija

Automatizovano testiranje u ovom projektu ima ulogu zavrsne potvrde da su backend ugovori, tokovi treninga i korisnicki scenariji na frontend strani zaista stabilni pri lokalnom pokretanju sistema. U skladu sa smernicama iz `PRD.md`, teziste verifikacije nije stavljeno na krhke pragove tacnosti pojedinacnih modela, vec na spolja vidljivo ponasanje sistema: ispravnost API odgovora, validaciju ulaza, doslednost Model Catalog-a, pravila asinhronog Training Job toka i korektan prikaz rezultata u korisnickom interfejsu. Takav pristup je prikladan za aplikaciju koja objedinjeno iznosi vise porodica klasifikatora, jer se stvarna pouzdanost sistema ne meri samo kroz konacnu predikciju, vec i kroz to da li su svi prateci tokovi medjusobno uskladjeni.

Za objedinjenu proveru u korenu repozitorijuma definisana je komanda `npm run test`, koja serijski poziva `npm run test:backend` i `npm run test:frontend`. Backend testovi se izvrsavaju kroz `pytest` nad direktorijumom `backend/tests`, dok se frontend scenariji pokrecu kroz `vitest` u okviru React aplikacije. Pored toga, komanda `npm run typecheck` koristi se za staticku proveru TypeScript sloja, kako bi se rano uocile neusaglasenosti izmedju tipova, stanja komponente i oblika podataka razmenjenih sa backend-om. [Tabela dopuniti: pregled glavnih test paketa, odgovarajucih scenarija i komandi za pokretanje.]

### Backend verifikacija

Backend testovi organizovani su oko stabilnih aplikacionih ugovora. U `backend/tests/test_health.py` proverava se da ruta `/api/health` vraca ispravan status servisa i da pri inicijalizaciji storage root-a zaista postoje direktorijumi `shipped-models`, `custom-models` i `registry`. Time se vec na pocetku suite-a potvrdjuje da aplikacija ispravno podize lokalno okruzenje nad kojim ce kasnije raditi i shipped i custom modeli.

Fajl `backend/tests/test_reference_prediction.py` pokriva inferencioni sloj na vise nivoa. Najpre se proverava referentni prototipski model nad kanonskim canvas ulazom, a zatim i artifact-backed scenariji za k-NN, SVM, Random Forest, MLP i CNN varijante. Ovakva struktura znaci da testovi ne proveravaju samo da li ruta `/api/predict` vraca neku cifru, vec i da backend ume da ucita odgovarajuci runtime artefakt, vrati uskladjene javne metapodatke i kod dubokih modela izlozi `deep_details` informacije koje frontend kasnije prikazuje u kartici detalja. U istom paketu proverava se i ruta `/api/models`, ukljucujuci slucajeve ostecenih ili nepotpunih shipped artefakata, kako bi kvar bio prijavljen jasno, a ne sakriven iza nekonzistentnog prikaza kataloga modela.

Validacija trening podataka izdvojena je u `backend/tests/test_training_csv_preview.py`. Ovi testovi potvrduju pozitivan scenario preview-a sa podrazumevanom podelom skupa, ali i dve vazne negativne grane: nekompatibilnu CSV semu i oznake klasa van opsega od 0 do 9. Time je verifikovan jedan od kljucnih granicnih uslova sistema, odnosno da korisnik ne moze pokrenuti Training Job nad datotekom koja ne odgovara MNIST-kompatibilnom formatu.

Najopsezniji backend paket je `backend/tests/test_custom_training.py`, jer obuhvata kompletan lifecycle custom modela. Testovi potvrdjuju da uspesan Training Job stvara novi unos u zajednickom katalogu, da se nad tim modelom moze odmah izvrsiti predikcija i da brisanje uklanja i javne metapodatke i runtime artefakte. Pored toga, proverena je podrska za sve podrzane porodice custom modela, ukljucujuci k-NN, SVM, Random Forest, MLP i CNN, kao i version-one ogranicenje da u jednom trenutku moze postojati samo jedan aktivan trening posao. Zavrsni scenario odbacivanja posla pri gasenju aplikacije dodatno potvrdjuje da prekinuti trening ne ostavlja polovicne ili lazne tragove u storage-u.

Poseban regresioni sloj predstavljaju `backend/tests/test_shipped_classical_training.py` i `backend/tests/test_shipped_deep_training.py`. Oni proveravaju da komponente za regeneraciju shipped modela iz repozitorijumskog `train.csv` zaista upisuju odgovarajuce JSON metapodatke, informacije o podeli skupa i runtime artefakte za klasicne i duboke modele. Time je verifikacija prosirena i na odrzavacke tokove projekta, a ne samo na krajnje korisnicke API pozive.

### Frontend verifikacija

Frontend verifikacija realizovana je u `frontend/src/App.test.tsx` kroz scenario-orijentisane testove koji simuliraju stvarno koriscenje aplikacije. Prvi skup testova proverava bootstrap aplikacije: pozive ka `/api/health` i `/api/models`, prikaz stanja spremnosti backend-a i inicijalno ucitavanje modela. Na to se nadovezuje test predikcije u kome korisnik boji canvas, bira model i salje zahtev na `/api/predict`, pri cemu se proveravaju i oblik poslatog payload-a i prikaz dobijenog rezultata.

Drugi skup testova pokriva prikaz i tumacenje Model Catalog-a. U tim scenarijima proveravaju se sortiranje leaderboard-a po razlicitim metrikama, selekcija reda u tabeli i prikaz detalja kao sto su dataset, vreme treniranja, konfuziona matrica i uzorci predikcija. Za duboke modele postoji i posebna provera da su `architecture_summary` i `epoch_curves` smesteni iza zasebnog taba, sto potvrdjuje da frontend ne samo prikazuje dodatne podatke, vec ih prikazuje na dosledan nacin koji ne remeti zajednicki pregled modela.

Trening rezim testiran je kao zaseban korisnicki tok. Testovi potvrdjuju da prelazak u training mode uklanja canvas i uvodi upload Training Dataset-a sa podrazumevanom podelom 80/10/10, da korisnik moze promeniti odnos split-a i da rezultat preview-a vraca ocekivane brojace primera. Poseban negativni scenario proverava da se backend greska iz CSV validacije prikazuje direktno u interfejsu, umesto da ostane sakrivena iza tihog neuspeha.

Za verziju jedan posebno su vazni regresioni testovi product policy pravila. Frontend testovi eksplicitno potvrdjuju da Custom Model mora imati rucno uneto ime bez automatskog predloga, kao i da neuspesan trening ostaje samo inline poruka za tekuci pokusaj i ne stvara neuspesan unos u leaderboard-u. Time je verifikacija prosirena i na poslovna pravila projekta, a ne samo na tehnicku ispravnost zahteva i odgovora. Dodatni scenario pokriva uspesan tok pokretanja custom treninga, pojavljivanje novog modela u zajednickom katalogu, njegov prikaz u detaljima i kasnije uklanjanje, cime je potvrdjen potpuni frontend-backend ciklus rada sa custom modelima.

### Opseg i ogranicenja verifikacije

Na osnovu dostupnih tragova u repozitorijumu moze se zakljuciti da je verifikacija usmerena na stabilnost ugovora, lifecycle modela i glavne korisnicke tokove, a ne na dokazivanje apsolutne superiornosti pojedinacnih klasifikatora. U tome se vidi doslednost sa planerskom odlukom iz `PRD.md` da prvi sloj testiranja treba da proverava spolja vidljivo ponasanje i ponovljivost, dok bi strogi kvalitetni pragovi po tacnosti bili krhkiji i manje prenosivi. Zbog toga u ovom radu nije opravdano unositi zbirne metrike pokrivenosti, benchmark rezultate ili rezultate dugotrajnih eksperimentalnih kampanja koje nisu trajno dokumentovane u samom repozitorijumu.

## Issue 11 - Zakljucak

Na osnovu prikazanog teorijskog okvira, projektovanja arhitekture i realizovane implementacije moze se zakljuciti da je osnovni cilj rada u najvecoj meri ostvaren. Izradjen je lokalno pokretljiv web sistem koji objedinjuje Python backend za validaciju podataka, inferenciju, trening i cuvanje modelskih artefakata, kao i React korisnicki interfejs namenjen crtanju cifre, izboru modela, pregledu rezultata i radu sa trening skupovima. Vrednost ovakvog resenja nije samo u tome sto omogucava klasifikaciju rukom pisanih cifara, vec i u tome sto vise porodica modela postavlja u zajednicki katalog i cini ih uporedivim kroz iste korisnicke tokove, metrike i detalje modela.

U odnosu na planirani obim rada, realizacija je ostala dosledna pocetnoj ideji da projekat bude istovremeno demonstracija masinskog ucenja i primer uredno organizovanog softverskog sistema. U repozitorijumu su zaista ostvareni jedinstveno pokretanje sistema iz korena projekta, zajednicki prikaz shipped i custom modela, predikcija nad cifrom nacrtanom na platnu, pregled metapodataka i metrika, CSV preview podele skupa i asinhroni tok prilagodjenog treniranja. Istovremeno, zavrsna verzija nije nekontrolisano sirila funkcionalnosti, vec je zadrzala jasne granice verzije jedan: jedan aktivan Training Job u datom trenutku, rucno imenovanje Custom Model-a, inline obradu neuspelog treninga bez trajnog neuspesnog unosa i jednu vidljivu startup tacku preko skripte `start.ps1`. Time je planirani obim rada preveden u realizaciju koja je pregledna, prenosiva i pogodna za demonstraciju na uobicajenoj Windows razvojnoj masini.

Sa obrazovnog i inzenjerskog stanovista, rad pokazuje da klasifikaciju slika nije dovoljno posmatrati samo kroz izbor algoritma. Jednaku vaznost imaju preprocessing ulaza, dosledna evaluacija, organizacija modelskih artefakata, jedinstven Model Catalog, smislen korisnicki interfejs i automatizovana verifikacija. Upravo povezivanje tih elemenata predstavlja najvazniji doprinos projekta, jer omogucava da razlike izmedju referentnog prototipa, klasicnih sklearn modela i dubokih PyTorch mreza budu vidljive ne samo u teorijskom opisu, vec i kroz konkretan rad jedne funkcionalne aplikacije.

Ipak, rezultat treba posmatrati u okviru ogranicenja prve verzije. Sistem je namenski ogranicen na lokalni rad, MNIST-kompatibilne CSV ulaze i projektno-lokalni storage, bez slozenijeg produkcionog okruzenja, vise istovremenih trening poslova ili sire podrske za druge tipove skupova podataka. Pored toga, verifikacija je usmerena pre svega na stabilnost ugovora i glavnih tokova rada, a ne na dugotrajne eksperimentalne kampanje ili formalne pragove pokrivenosti. Ovakva ogranicenja ne umanjuju vrednost rada, ali jasno pokazuju da je realizovana disciplinovano zaokruzena verzija jedan, a ne potpuno opsti sistem za upravljanje modelima.

Na toj osnovi mogu se izdvojiti sledeci pravci daljeg razvoja:

1. Prosirenje orkestracije treninga tako da podrzi red cekanja, vise poslova ili nastavak prekinute obuke, uz zadrzavanje jasnih pravila evidencije i perzistencije.
2. Prosirenje ingest i evaluacionog sloja na sire skupove podataka i dodatne rezime rezultate, uz ocuvanje kanonskog preprocessing ugovora.
3. Uvodjenje jednostavnijeg nacina distribucije i pokretanja van lokalnog Windows scenarija, na primer kroz dodatnu instalacionu ili kontejnersku varijantu.
4. Sprovodjenje dubljih uporednih eksperimenata i bogatijih interpretacionih priloga koji bi omogucili jos preciznije tumacenje razlika medju modelima.

## Issue 12 - Literatura

Poglavlje o literaturi u ovom radu ima dvostruku ulogu. Sa jedne strane, ono obezbedjuje teorijski oslonac za objasnjenje klasifikacije rukom pisanih cifara, primenjenih algoritama i evaluacionih metrika. Sa druge strane, ono razdvaja strucne izvore od tehnicke dokumentacije i internih projektnih dokumenata koji su korisceni da se dosledno opise realizovana aplikacija. Takvo razdvajanje je posebno vazno u ovom projektu, jer rad istovremeno obuhvata teorijske osnove masinskog ucenja i konkretan softverski sistem zasnovan na Python backend-u, React frontend-u, biblioteci scikit-learn za klasicne modele i biblioteci PyTorch za duboke modele.

### Strucna literatura za algoritme i biblioteke

Za deo rada koji se odnosi na klasicne metode masinskog ucenja, preprocessing tokove i upotrebu biblioteke scikit-learn, osnovni strucni oslonac predstavlja knjiga Sebastiana Raschke i saradnika, _Machine Learning with PyTorch and Scikit-Learn_. Ovaj izvor je pogodan zato sto na jednom mestu povezuje teorijski i implementacioni prikaz klasicnih klasifikatora, rada sa skupovima podataka, transformacija osobina i standardnih evaluacionih postupaka. U okviru ovog rada takva literatura je relevantna pre svega za objasnjenje k-NN, linearne SVM, ansamblskog pristupa Random Forest, logike preprocessing koraka i tumacenja klasifikacionih metrika koje su prisutne i u samom backend-u. [Literatura dopuniti: uneti puno ime svih autora, izdanje, godinu i izdavaca prema skolskom formatu.]

Za deo rada koji pokriva neuronske mreze i duboko ucenje, glavni izvor predstavlja knjiga Eli Stevens, Luca Antiga i Thomas Viehmann, _Deep Learning with PyTorch_. Ovaj naslov je neposredno koristan za teorijsko obrazlozenje rada sa tenzorima, optimizacionih postupaka, gubitackih funkcija i arhitektura namenjenih klasifikaciji slika. U kontekstu ovog projekta on podrzava objasnjenje MLP i CNN modela, odnosno razumevanje zasto se potpuno povezane i konvolucione mreze razlicito ponasaju nad MNIST ulazom i zasto PyTorch predstavlja odgovarajuci izbor za realizaciju shipped i custom deep modela. [Literatura dopuniti: uneti godinu, izdanje i izdavaca prema skolskom formatu.]

### Izvor podataka

Kao izvor podataka za eksperimentalni i implementacioni deo rada koristi se MNIST-kompatibilan CSV format, pri cemu repozitorijum i backend logika eksplicitno polaze od strukture "label + 784 pixel vrednosti". Za formalno navodjenje porekla takvog skupa podataka u radu treba koristiti Kaggle stranicu sa koje je preuzet CSV zapis nad kojim su obavljeni treniranje, preview podele skupa i evaluacija shipped modela. Navodjenje Kaggle izvora je vazno ne samo zbog akademske korektnosti, vec i zbog toga sto pojasnjava zasto se u projektu radi sa tabelarnim CSV reprezentacijama, a ne sa originalnim IDX binarnim fajlovima. [Literatura dopuniti: upisati tacan naziv Kaggle skupa, URL i datum pristupa.]

### Tehnicka dokumentacija i interni projektni izvori

Pored navedene strucne literature, u radu je koriscena i tehnicka dokumentacija biblioteka i alata koji cine realizovani sistem. Iz zavisnosti navedenih u `requirements.txt` i iz arhitekture opisane u ostatku rada vidi se da posebnu ulogu imaju scikit-learn i PyTorch kao biblioteke za modele, FastAPI i Uvicorn kao backend servisni sloj, kao i React i Vite na frontend strani. Takvi izvori nisu zamena za naucnu ili strucnu literaturu, ali su korisni kada je potrebno precizno objasniti API ugovore, razvojno okruzenje, pokretanje sistema ili ponasanje konkretnih biblioteka u implementacionom delu rada. [Literatura dopuniti: po potrebi uneti zvanicne URL adrese dokumentacije i datum pristupa.]

Internu projektnu dokumentaciju cine `PRD.md`, `CONTEXT.md` i `docs/README.md`. Ovi fajlovi ne treba da budu tretirani kao spoljna naucna literatura, ali imaju vrednost pomocnih izvora za dosledno predstavljanje opsega projekta, terminologije modelskog kataloga, pravila version-one implementacije i odnosa izmedju backend i frontend sloja. Zbog toga ih je svrsishodno koristiti kao unutrasnji oslonac pri pisanju rada, dok se u samoj formalnoj bibliografiji jasno odvaja literatura spoljnog porekla od repozitorijumskih dokumenata.

Na osnovu ovakve podele moguce je formirati preglednu i metodoloski korektnu literaturu rada: strucni izvori za teoriju i algoritme, zasebno naveden izvor podataka i odvojeno tehnicka dokumentacija koja podupire opis realizacije sistema. Pri finalnoj redakturi potrebno je sve navedene jedinice prevesti u bibliografski standard koji skola zahteva i proveriti da svaki teorijski navod iz prethodnih poglavlja ima jasan i proverljiv izvor.

## Issue 13 - Prilozi

### Opis

- Zavrsni dodaci okupljaju materijal koji dopunjuje glavni tekst bez opterecivanja osnovnih poglavlja.

### Treba da pokrije

- Screenshot-ove interfejsa i glavnih ekrana.
- Primere ulaznih i izlaznih JSON struktura ili metapodataka modela.
- Kratak prilog sa nacinom pokretanja sistema i organizacijom podataka.
- Po potrebi odabrane isecke koda ili prikaz registra modela.

### Repo tragovi

- `start.ps1`
- `data/registry/models.json`
- `data/shipped-models/reference-prototype-v1.json`
- `frontend/src/App.tsx`

### Treba dopisati

- [ ] Dodati slike interfejsa i po potrebi anotacije.
- [ ] Izabrati koje API primere i koje isecke koda vredi staviti u prilog.
- [ ] Proveriti da prilozi dopunjuju rad, a ne ponavljaju glavna poglavlja.
