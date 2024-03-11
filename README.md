# SSB pre-system
Forsystem for månedlige og kvartalsvise NR-statistikker

[![PyPI](https://img.shields.io/pypi/v/ssb-pre-system.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/ssb-pre-system.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/ssb-pre-system)][pypi status]
[![License](https://img.shields.io/pypi/l/ssb-pre-system)][license]

[![Documentation](https://github.com/statisticsnorway/ssb-pre-system/actions/workflows/docs.yml/badge.svg)][documentation]
[![Tests](https://github.com/statisticsnorway/ssb-pre-system/actions/workflows/tests.yml/badge.svg)][tests]
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=statisticsnorway_ssb-pre-system&metric=coverage)][sonarcov]
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=statisticsnorway_ssb-pre-system&metric=alert_status)][sonarquality]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)][poetry]

[pypi status]: https://pypi.org/project/ssb-pre-system/
[documentation]: https://statisticsnorway.github.io/ssb-pre-system
[tests]: https://github.com/statisticsnorway/ssb-pre-system/actions?workflow=Tests

[sonarcov]: https://sonarcloud.io/summary/overall?id=statisticsnorway_ssb-pre-system
[sonarquality]: https://sonarcloud.io/summary/overall?id=statisticsnorway_ssb-pre-system
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[poetry]: https://python-poetry.org/

## Features

Prosjektet består av to klasser (**Formula** og **PreSystem**) og to funksjoner (**convert** og **convert_step**).

### Formula

**Indicator**-underklassen definerer et indikatorobjekt som favner om de fleste indikatorer i nasjonalregnskapet,

$$
  x_t = x_T\cdot\frac{k_t\sum_i w_{i,T} I_{i,t}}{\sum_{s\in T}k_s\sum_i w_{i,T} I_{i,s}},
$$

der $x$ er den aktuelle nasjonalregnskapsvariabelen, $w$ er vekter, $I$ er indikatorer. $T$ betegner basisåret.
$k$ er en korreksjon som er lik én med mindre brukeren ønsker å foreta en korreksjon.

**FDeflate**-underklassen tar utgangspunkt i en eksisterende formel (for eksempel en **Indicator**-instans) og deflaterer denne,

$$
  \sum_{s\in T}x_s\cdot\frac{k_t x_t/\sum_i w_{i,T} I_{i,t}}{\sum_{s\in T}k_s x_s/\sum_i w_{i,T} I_{i,s}}.
$$

**FInflate**-underklassen tar utgangspunkt i en eksisterende formel (for eksempel en **Indicator**-instans) og inflaterer denne,

$$
  \sum_{s\in T}x_s\cdot\frac{k_t x_t\sum_i w_{i,T} I_{i,t}}{\sum_{s\in T}k_s x_s\sum_i w_{i,T} I_{i,s}}.
$$

**FSum** summerer andre **Formula**-insanser, **FSumProd** lager et summerprodukt, **FMutlt** multipliserer to instanser, og **FDiv** dividerer.

Alle undeklassene har metodene **what** og **evaluate**. `formel.what` vil returnere en tekstlig representasjon av definisjonen på formelen. Dette lar brukeren spore seg tilbake til én eller flere **Indicator**-instanser (alle formler må til slutt ende i **Indicator**-instanser). `formel.evaluate(års_df, indikator_df, vekt_df, korreksjon_df)` returnerer en **Pandas**-serie som er den aktuelle formelen evaluert gjenstand for data.

### PreSystem
Klassen **PreSystem** lar brukeren initialisere et forsystem-objekt. Dette har som oppgave å holde instanser av **Formel**-objekter og la brukeren enkelt evaluere alle formler som er en del av forsystemet.

### Convert og convert_step
Dette er funksjoner som lar brukeren konvertere en **Pandas** **DataFrame** fra én frekvens til en annen.
## Installation

You can install _SSB pre-system_ via [pip] from [PyPI]:

```console
pip install ssb-pre-system
```

## Usage

Please see the [Reference Guide] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_SSB pre-system_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [Statistics Norway]'s [SSB PyPI Template].

[statistics norway]: https://www.ssb.no/en
[pypi]: https://pypi.org/
[ssb pypi template]: https://github.com/statisticsnorway/ssb-pypitemplate
[file an issue]: https://github.com/statisticsnorway/ssb-pre-system/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/statisticsnorway/ssb-pre-system/blob/main/LICENSE
[contributor guide]: https://github.com/statisticsnorway/ssb-pre-system/blob/main/CONTRIBUTING.md
[reference guide]: https://statisticsnorway.github.io/ssb-pre-system/reference.html
