# TFG Radiomics

Repositorio para el procesamiento longitudinal de imagen médica, extracción de radiomics y entrenamiento de un MLP sobre variables de cambio longitudinal.

## Estructura

```text
.
├── data/
│   ├── base/
│   └── raw/
├── outputs/
├── scripts/
├── src/
├── requirements.txt
└── README.md
```

## Capas del pipeline

### Datos base

- `src/data/ingest/`: reorganización física del dataset en `data/raw`
- `src/data/indexing/`: indexado y validación tabular
- `src/preprocessing/build_longitudinal_pairs.py`: construcción de pares longitudinales en formato `long`

Artefactos principales:

- `outputs/indexing/index.csv`
- `outputs/indexing/validated_index.csv`
- `outputs/pairs/longitudinal_pairs.csv`

### Radiomics base

- `src/radiomics/extract_features.py`

Artefacto principal:

- `outputs/features/longitudinal_features_long.csv`

### Features para modelado

- `scripts/prepare_modeling_features.py`: prepara la capa de features de modelado
- `scripts/features/prepare_features_uid.py`: construye una matriz por par longitudinal
- `scripts/features/prepare_delta_features.py`: calcula `delta_features_uid.csv`

Artefactos principales:

- `outputs/features/features_uid.csv`
- `outputs/features/delta_features_uid.csv`

### Estabilidad radiomics

- `scripts/export_icc_subset.py`: exporta el subset fijo de baseline usado para ICC
- `src/radiomics/icc.py`: calcula ICC y filtra features robustas

Artefactos principales:

- `outputs/features/icc_subset_keys.csv`
- `outputs/features/icc_results.csv`
- `outputs/features/icc_features_keep.csv`
- `outputs/features/icc_features_discard.csv`

### Entrenamiento

- `scripts/train_mlp.py`

Entradas:

- `outputs/features/delta_features_uid.csv`
- `outputs/features/icc_features_keep.csv`
- `outputs/review/classification_outcomes.csv`

Salidas:

- `outputs/models/mlp/<timestamp>/model.joblib`
- `outputs/models/mlp/<timestamp>/metrics.json`
- `outputs/models/mlp/<timestamp>/train_uids.csv`
- `outputs/models/mlp/<timestamp>/test_uids.csv`

## Ejecución

Pipeline completo:

```powershell
python scripts/run_full_pipeline.py
```

Opciones útiles:

```powershell
python scripts/run_full_pipeline.py --skip-data
python scripts/run_full_pipeline.py --skip-icc
python scripts/run_full_pipeline.py --generate-review-wide
```

Ejecución manual por etapas:

```powershell
python scripts/export_icc_subset.py
$env:ICC_REEXTRACT="1"
python src/radiomics/icc.py
python scripts/prepare_modeling_features.py
python scripts/train_mlp.py
```

## Representaciones usadas

El proyecto trabaja con dos vistas oficiales del mismo problema:

- `long`: para pares longitudinales y radiomics por timepoint
- `pairwise`: para modelado supervisado por par longitudinal

Esto permite mantener un pipeline limpio para preparación y, a la vez, una capa específica para el entrenamiento del MLP.
