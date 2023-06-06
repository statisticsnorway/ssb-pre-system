# nr-forsystem

Forsystem for månedlige og kvartalsevise NR-statistikker

Opprettet av:
Magnus Kvåle Helliesen <magnus.helliesen@gmail.com>

---

Prosjektet består av to klasser (**Formula** og **PreSystem**) og to funksjoner (**convert** og **convert_step**).

## Formula
Klassen **Formula** består av en rekke underklasser: **Indicator**, **FDeflate**, **FInflate**, **FSum**, **FSumProd**, **FMutl** og **FDiv**.

**Indicator**-underklassen definerer et indikatorobjekt som favner om de fleste indikatorer i nasjonalregnskapet,

$$
  x_t = x_T\cdot\frac{k_t\sum_i w_{i,T} I_{i,t}}{\sum_{s\in T}k_s\sum_i w_{i,T} I_{i,s}},
$$

der $x$ er den aktuelle nasjonalregnskapsvariabelen, $w$ er vekter, $I$ er indikatorer. $T$ betegner basisåret.
$k$ er en korreksjon som er lik én med mindre brukeren ønsker å foreta en korreksjon

**FDeflate**-underklassen tar utgangspunkt i en eksisterende formel (for eksempel en **Indicator**-instans) og deflaterer denne,

$$
  \sum_{s\in T}x_t\cdot\frac{k_t x_t/\sum_i w_{i,T} I_{i,t}}{\sum_{s\in T}k_s x_s/\sum_i w_{i,T} I_{i,s}}.
$$

**FInflate**-underklassen tar utgangspunkt i en eksisterende formel (for eksempel en **Indicator**-instans) og inflaterer denne,

$$
  \sum_{s\in T}x_t\cdot\frac{k_t x_t\sum_i w_{i,T} I_{i,t}}{\sum_{s\in T}k_s x_s\sum_i w_{i,T} I_{i,s}}.
$$

**FSum** summerer andre **Formula**-insanser, **FSumProd** lager et summerprodukt, **FMutlt** multipliserer to instanser, og **FDiv** dividerer.

Alle undeklassene har metodene **what** og **evaluate**. `formel.what` vil returnere en tekstlig representasjon av definisjonen på formelen. Dette lar brukeren spore seg tilbake til én eller flere **Indicator**-instanser. `formel.evaluate(års_df, indikator_df, vekt_df, korreksjon_df)` returnerer en **Pandas**-serie som er den aktuelle formelen evaluert gjenstand for data.

## PreSystem
Klassen **PreSystem** lar brukeren initialisere et forsystem-objekt. Dette har som oppgave å holde instanser av **Formel**-objekter og la brukeren enkelt evaluere alle formler som er en del av forsystemet

## Convert og convert_step
Dette er funksjoner som lar brukeren konvertere en **Pandas** **DataFrame** fra én frekvens til en annen.
