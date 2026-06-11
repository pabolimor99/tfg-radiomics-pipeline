# TFG Radiomics

Repositorio para procesamiento longitudinal de imagen mádica, extraccion de
radiomics, seleccion de variables robustas y entrenamiento de modelos
supervisados sobre cambios longitudinales.

## Estructura

`src/` contiene la logica reutilizable del proyecto. `scripts/` contiene solo
wrappers de ejecucion que llaman a funciones de `src/`.

```text
.
|-- data/                 # datos brutos/reorganizados
|-- outputs/
|   |-- indexing/         # indices y validacion tabular
|   |-- pairs/            # pares longitudinales
|   |-- features/         # features derivadas
|   |-- review/           # etiquetas/revision manual
|   `-- models/           # modelos y busquedas
|-- figures/              # figuras de resultados
|-- scripts/
|   |-- data/             # comandos de preparacion de data/raw
|   |-- features/         # comandos de preparacion de features
|   |-- models/           # comandos de entrenamiento final
|   |-- optuna/           # comandos de busqueda Optuna
|   `-- reporting/        # comandos de resumen y figuras
|-- src/
|   |-- data/
|   |-- preprocessing/
|   |-- radiomics/
|   |-- features/
|   |-- modeling/
|   |-- reporting/
|   `-- review/
|-- requirements.txt
`-- README.md
```

## Pipeline

### 1. Indexado y validacion

- `src/data/prepare_raw.py`: reconstruye `data/raw` desde `data/base`.
- `src/data/ingest/`: reorganizacion fisica del dataset en `data/raw`.
- `src/data/indexing/indexer.py`: genera el indice inicial.
- `src/data/indexing/validator.py`: valida pares imagen-mascara.

Artefactos principales:

- `outputs/indexing/index.csv`
- `outputs/indexing/validated_index.csv`
- `outputs/indexing/invalid_pairs.csv`

### 2. Pares longitudinales

- `src/preprocessing/build_longitudinal_pairs.py`

Artefacto principal:

- `outputs/pairs/longitudinal_pairs.csv`

### 3. Radiomics

- `src/radiomics/extract_features.py`: extrae radiomicas.
- `src/radiomics/icc.py`: calcula estabilidad/ICC.

Artefactos principales:

- `outputs/features/longitudinal_features_long.csv`
- `outputs/features/icc_results.csv`
- `outputs/features/icc_perturbed_features.csv`

Los CSV de `outputs/features/` se mantienen en el repositorio porque son
costosos de recalcular.

### 4. Features de modelado

Logica:

- `src/features/prepare_features_uid.py`
- `src/features/prepare_delta_features.py`
- `src/features/prepare_modeling_features.py`

Comando:

- `scripts/features/prepare_modeling_features.py`

Artefactos principales:

- `outputs/features/features_uid.csv`
- `outputs/features/delta_features_uid.csv`

### 5. Modelado

El criterio por defecto del proyecto es `balanced_accuracy`.

Logica:

- `src/modeling/train.py`
- `src/modeling/feature_selection.py`
- `src/modeling/models/lda.py`
- `src/modeling/models/mlp.py`
- `src/modeling/models/random_forest.py`
- `src/modeling/models/lgbm.py`
- `src/modeling/models/xgboost.py`

Comandos:

- `scripts/models/train_lda.py`
- `scripts/models/train_mlp.py`
- `scripts/models/train_random_forest.py`
- `scripts/models/train_lgbm.py`
- `scripts/models/train_xgboost.py`

Salidas principales:

- `outputs/models/<modelo>/<timestamp>/model.joblib`
- `outputs/models/<modelo>/<timestamp>/metrics.json`
- `outputs/models/<modelo>/<timestamp>/test_predictions.csv`

### 6. Optuna

La rama oficial de busqueda es BACC.

Logica:

```text
src/modeling/optuna/
|-- train.py
|-- config.py
|-- search_spaces/
|   |-- lda.py
|   |-- mlp.py
|   |-- random_forest.py
|   |-- lgbm.py
|   `-- xgboost.py
`-- bacc/
    |-- lda.py
    |-- mlp.py
    |-- random_forest.py
    |-- lgbm.py
    `-- xgboost.py
```

Comandos:

```powershell
python -m scripts.optuna.bacc.lda
python -m scripts.optuna.bacc.mlp
python -m scripts.optuna.bacc.random_forest
python -m scripts.optuna.bacc.lgbm
python -m scripts.optuna.bacc.xgboost
```

### 7. Reporting

Logica:

- `src/reporting/summarize_optuna_bacc.py`
- `src/reporting/plot_final_model_results.py`
- `src/reporting/plot_feature_robustness_discrimination.py`

Comandos:

```powershell
python -m scripts.reporting.summarize_optuna_bacc
python -m scripts.reporting.plot_final_model_results
python -m scripts.reporting.plot_feature_robustness_discrimination
```

## Ejecucion manual

### Preparacion del entorno

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Si se van a usar los CSV ya calculados de `outputs/features/`, basta con
instalar las dependencias de modelado:

```powershell
python -m pip install -r requirements-modeling.txt
```

### Dependencias radiomicas

Para recalcular radiomicas desde imagenes hace falta instalar tambien
`pyradiomics` y `SimpleITK`.

En Windows, `pyradiomics` puede requerir Microsoft C++ Build Tools. Si aparece
un error como `Microsoft Visual C++ 14.0 or greater is required`, instalar las
Build Tools desde PowerShell como administrador:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Despues de la instalacion, cerrar PowerShell, abrir una nueva terminal, activar
la venv otra vez y ejecutar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==1.26.4
python -m pip install -r requirements-radiomics.txt --no-build-isolation
```

Es importante instalar o actualizar `wheel` antes de instalar `pyradiomics`;
si no, puede aparecer el error `invalid command 'bdist_wheel'`.

Comprobacion rapida:

```powershell
python -c "import numpy, radiomics, SimpleITK, sklearn; print('OK')"
```

No usar `python -m pip install -r requirements.txt` para resolver el entorno en
Windows si se va a compilar `pyradiomics`; usar los ficheros separados
`requirements-modeling.txt` y `requirements-radiomics.txt`.

### Workers en extraccion radiomics

La extraccion de caracteristicas radiomicas puede paralelizarse con la variable
de entorno `RADIOMICS_WORKERS`. Por defecto el script usa un valor conservador,
como maximo 2 workers:

```python
DEFAULT_WORKERS = max(1, min(2, (os.cpu_count() or 4) - 1))
```

Para aumentar el paralelismo en PowerShell:

```powershell
$env:RADIOMICS_WORKERS = "4"
python -m src.radiomics.extract_features
```

Para usar otro valor:

```powershell
$env:RADIOMICS_WORKERS = "6"
python -m src.radiomics.extract_features
```

Se recomienda empezar con 4 workers y subir progresivamente. La extraccion
radiomics puede consumir bastante memoria, por lo que un numero alto de workers
puede saturar la RAM antes que mejorar el tiempo de ejecucion.

Para volver al comportamiento por defecto en la misma terminal:

```powershell
Remove-Item Env:RADIOMICS_WORKERS
```

Preparar `data/raw` desde `data/base/inputsTr` y `data/base/targetsTr`:

```powershell
python -m scripts.data.prepare_raw
```

Si `data/raw` ya existe y se quiere reconstruir desde cero:

```powershell
python -m scripts.data.prepare_raw --clean
```

Preparar features de modelado:

```powershell
python -m scripts.features.prepare_modeling_features
```

Entrenar modelos finales:

```powershell
python -m scripts.models.train_lda
python -m scripts.models.train_mlp
python -m scripts.models.train_random_forest
python -m scripts.models.train_lgbm
python -m scripts.models.train_xgboost
```

## Representaciones

El proyecto trabaja con dos vistas:

- `long`: pares y radiomicas por timepoint.
- `pairwise`: matriz por par longitudinal.

La vista `pairwise` se deriva desde
`outputs/features/longitudinal_features_long.csv`, sin reextraer radiomics.