# BloMLster

#Intro

Målet med prosjektet er at min bestemor som elsker blomster og er redd for teknologi skal kunne overkomme denne frykten ved å bruke det til å leke med blomster. 

## Mappestruktur

```
ml-prosjekt/
├── data/
│   └── data/           # Data fra tredjepartskilder
│        └──flowers/     
├── notebooks/          # Jupyter notebooks 
│        └──01trening/
│        └──02evaluering/        
├── src/              # Gjenbrukbar Python-kode (config, hjelpefunksjoner)
├── models/           # Trente/lagrede modeller
├── reports/
│   └── figures/      # Genererte figurer
├── requirements-rocm.txt
├── requirements.txt
├── .env.example
├──app.py
├──BloMLster_rapport
└── README.md
```

## Kom i gang

```bash
# Opprett og aktiver virtuelt miljø
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# velg en av de 2 neste basert på din hardware

# for Nvidia GPU
pip install -r requirements.txt

#for AMD GPU
py -3.13 -m venv .venv-rocm
.venv-rocm\Scripts\python -m pip install -r requirements-rocm.txt
.venv-rocm\Scripts\rocm-sdk init

# Registrer miljøet som Jupyter-kernel
python -m ipykernel install --user --name ml-prosjekt

# Start Jupyter
jupyter lab
```


## Bruke `src/` i notebooks

```python
import sys; sys.path.append("..")
from src.config import RAW_DIR, PROCESSED_DIR, SEED
from src.utils import set_seed, save_fig

set_seed()
df = pd.read_csv(RAW_DIR / "data.csv")
```

## Data

Hvordan få tak i data:

last ned zip fil fra:
https://www.kaggle.com/datasets/supriyoain/flower-classification?resource=download
pakk ut og legg rot mappen i data. (data/flowers/artene)


## Starte appen
kjør app.py
følg lenken i terminalen

## Resultater
[Rapport](BloMLster_rapport.md)
accuracy: 92.4%

macro_f1: 91.6%

top3: 97.9%

![bilde av nettsiden](web.png)


## Kilder

torch for kode og inspirasjon https://docs.pytorch.org/tutorials/beginner/basics/intro.html

drøfting av strategier, kode og skriving claude code

