# Maparea categoriilor furnizorului pe ale noastre

Mapează cele 3.030 de valori de categorie ale furnizorului (2.678 după ignorarea literelor
mari/mici și a spațiilor) pe arborele nostru de 150 de categorii. Livrabil: `output/mapping.csv`.

## Rulare

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # cele 4 CSV-uri de intrare (necomise în repo) se pun în rădăcină
    python build_mapping.py    # output/mapping.csv + diagnostics.csv + product_predictions.csv
    python analysis.py         # output/cost_of_errors.csv + weak_spots.csv
    python -m pytest -v        # 3 teste de format pe mapping.csv

`exploration/` conține experimentele (trăsături, pack_size, ponderi de vot, verificări manuale);
se rulează din rădăcină, ex. `python exploration/vote_cv.py`.

## Ce am făcut

Numele categoriilor furnizorului sunt abreviate („Spr", „Be", „tlt ro"), deci am judecat după
produse. Pașii:
1. Cheie de categorie = litere mici, fără spații; valorile cu aceeași cheie primesc același răspuns.
2. Clasificator (TF-IDF pe nume + brand, n-grame de cuvinte și caractere, plus `pack_size` parsat
   în unitate / mărime / număr de bucăți; regresie logistică) antrenat pe cele 4.018 produse
   etichetate manual.
3. Fiecare produs primește o categorie prezisă; fiecare categorie a furnizorului primește
   destinația cu cele mai multe voturi (un produs = un vot; produsele etichetate contează sigur).
4. O linie în CSV pentru fiecare valoare scrisă original.

## Dovezi și verificare

- Validare încrucișată pe grupuri (categorii întregi scoase la test, ca să simuleze cele 1.051
  fără etichete): pe produs 0,953; **pe categorie 0,882; ponderat cu venit 0,897**. Se evaluează
  doar produsele etichetate, deci scorul e zgomotos; diferențele sub ~0,5 puncte nu contează.
- Înainte/după, `pack_size`: 0,870 → 0,878 pe categorie (0,908 → 0,920 ponderat cu venit).
- Înainte/după, regula de vot: cu votul pe venit, `citr.frt` mergea în Berries din cauza unui
  singur produs de 94k €; cu votul pe număr merge în Citrus fruit. 5 variante de vot dau
  0,875–0,882 pe categorie (în zgomot), deci alegerea e o decizie de judecată, nu una dovedită.
- 3 teste automate: o linie pentru fiecare valoare distinctă, aceeași cheie => același răspuns,
  toate ID-urile există în arborele nostru.
- Am citit manual produsele celor mai riscante categorii.

## Unde e cel mai slab și ce ne-ar costa

- Multe categorii riscante sunt **amestecuri reale** (nicio destinație unică nu e corectă):
  `dog-fd` (mâncare câini + scutece), `hot/tea` (ceai + apă), `bee/ale.st` (bere + chipsuri),
  `shwrbath` (gel de duș + fructe tropicale), `smkdfish` (pește afumat + congelat). Altele sunt
  greșeli ale modelului (ex. `GRAPEFRUIT` prezis Berries).
- Cu `sales.csv`: 107 M din 358 M EUR (29,9%) sunt la produse a căror categorie prezisă diferă de
  destinația categoriei lor. E o **limită superioară** (amestecuri + zgomot de model) și nu o pot
  reconcilia cu ~10% din validare. Costul e împrăștiat: primele 20 de categorii dețin 11%.
- 583 de categorii fără nicio etichetă și amestecate (118 M EUR venit total): prioritate pentru
  revizuire umană (`output/weak_spots.csv`).
- Votul pe număr de produse înseamnă că venitul nu influențează destinația, doar raportarea
  costului. Cu votul pe venit s-ar schimba ~300 de categorii.
- 6 categorii ale noastre nu primesc nicio mapare (Bananas, Jams & sweet spreads, Fresh fish,
  Burgers & nuggets, Stock & gravy, Lemonade & mixers). O căutare pe cuvinte-cheie nu a găsit
  categorii ale furnizorului care să le aparțină clar; nu le-am forțat.

## Ce nu am făcut, intenționat

Fără LLM/embeddings (eroarea vine în mare parte din amestecuri); fără împărțirea categoriilor
amestecate (se cere o singură destinație); fără revizuirea tuturor celor 2.678 de categorii;
fără tuning de hiperparametri.

## Folosirea AI

Am folosit Claude ca asistent: pentru idei de abordare (clasificare pe produs + vot pe categorie, parsarea `pack_size`) 
și pentru a redacta codul . Am rulat eu fiecare pas, am verificat manual produsele din categoriile riscante 
(`dog-fd`, `hot/tea`, `citr.frt`, `bee/ale.st` etc.).

## Găsit în afara task-ului, nerezolvat

- 2.398 de produse au venit 0.
- 299 din 1.627 de categorii cu etichete sunt amestecate: categorisirea furnizorului e impură.
- Aceeași familie apare în mai multe categorii (`tinned-to`, `tinnefrt`), iar regula furnizorului
  (litere/spații) nu le unește.
- Categorii promoționale (`promowkq3`, „Launches Oct") nu pot fi mapate semnificativ.