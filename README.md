# 🚗 Toyota Corolla ML Pipeline — Parcial Inteligencia Empresarial y Datamining

¡Bienvenido! Este repo contiene el **pipeline de ingestión, limpieza y modelado** del histórico de precios del Toyota Corolla, orquestado con **Dagster** y trackeado en **MLflow**.  
Sigue los pasos rápidos y en minutos tendrás los _assets_ materializados en tu máquina 🤖.

---

## 🌟 Paso a Paso

```bash
git clone https://github.com/alvarotoledo01/parcial_toyotaCorolla.git
cd parcial_toyotaCorolla

# 1️⃣  Entorno Conda
conda create --name pazposse_toledo python=3.11 -y
conda activate pazposse_toledo

# 2️⃣  Dependencias con Poetry
pip install poetry                      # se instala sólo dentro del env
poetry install --no-root                # baja EXACTAMENTE lo del lock

# 3️⃣  Variables de entorno (Dagster & MLflow)
echo "MLFLOW_TRACKING_URI=https://8270-186-122-11-149.ngrok-free.app/
MLFLOW_EXPERIMENT_NAME=toyota_parcial" > .env

# 4️⃣  ¡Despega!
dagster dev                             # abre 127.0.0.1:3000
```

## Estructura del Proyecto

```bash
parcial_toyotaCorolla/
├─ dagster_project/          # pipelines, assets, IO managers
├─ data/                     # datasets & scripts de descarga
├─ notebooks/                # exploración y EDA
├─ pyproject.toml            # dependencias (poetry)
├─ poetry.lock               # versiones exactas
├─ environment.yml           # tu conda minimal
└─ README.md                 # you're here 🚀
```
