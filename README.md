# ML-prosjekt

Kort beskrivelse av prosjektet: hva skal predikeres/analyseres, og hvorfor.

Målet med prosjektet er at min bestemor som elsker blomster og er redd for teknologi skal kunne bruke denne modell på en nettside på telefon peke på en blome i huset hennes og den skal bli gjenkjent

## Mappestruktur

```
ml-prosjekt/
├── data/
│   ├── raw/          # Originaldata – aldri endre disse filene
│   ├── interim/      # Mellomsteg i rensing/transformasjon
│   ├── processed/    # Ferdig data klar for modellering
│   └── external/     # Data fra tredjepartskilder
├── notebooks/        # Jupyter notebooks (se navnekonvensjon under)
├── src/              # Gjenbrukbar Python-kode (config, hjelpefunksjoner)
├── models/           # Trente/lagrede modeller
├── reports/
│   └── figures/      # Genererte figurer
├── requirements.txt
├── .env.example
└── README.md
```

## Kom i gang

```bash
# Opprett og aktiver virtuelt miljø
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Installer avhengigheter
pip install -r requirements.txt

# Registrer miljøet som Jupyter-kernel
python -m ipykernel install --user --name ml-prosjekt

# Start Jupyter
jupyter lab
```

## Navnekonvensjon for notebooks

`{nr}_{beskrivelse}.ipynb`, f.eks.:

- `01_eda.ipynb` – utforskende dataanalyse
- `02_preprocessing.ipynb` – rensing og feature engineering
- `03_modeling.ipynb` – trening og evaluering

## Bruke `src/` i notebooks

```python
import sys; sys.path.append("..")
from src.config import RAW_DIR, PROCESSED_DIR, SEED
from src.utils import set_seed, save_fig

set_seed()
df = pd.read_csv(RAW_DIR / "data.csv")
```

## Data

Beskriv datakilde(r), hvordan de hentes, og eventuelle lisenser.
lastet ned zip fil fra:
https://www.kaggle.com/datasets/supriyoain/flower-classification?resource=download


## Resultater

Oppsummer viktigste funn og modellytelse her.
