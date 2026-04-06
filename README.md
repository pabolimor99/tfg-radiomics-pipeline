# TFG Radiomics

Este repositorio contiene el código y la organización de datos utilizados para el desarrollo del TFG, centrado en el análisis de imágenes médicas longitudinales y la extracción de características radiómicas.

## Estructura general del proyecto

```text
.
├── data/
│   ├── base/
│   └── raw/
├── src/
├── scripts/
├── outputs/
├── requirements.txt
└── README.md

```

## Descripción de carpetas

### `data/`
Contiene los datos del proyecto.

- **`data/base/`**  
  Incluye los datos originales del dataset, tal y como se descargaron inicialmente.

- **`data/raw/`**  
  Contiene una reorganización inicial de los datos originales, con una estructura más cómoda para trabajar durante el desarrollo del pipeline.

### `src/`
Incluye la lógica principal del proyecto. Aquí se definen las funciones y módulos reutilizables para tareas como carga de datos, validación, preprocesado y extracción de características.

### `scripts/`
Contiene los scripts ejecutables del proyecto. Estos scripts utilizan las funciones definidas en `src/` para realizar tareas concretas, como reorganizar datos, generar tablas auxiliares o lanzar la extracción de radiomics

### `outputs/`
Almacena los resultados generados durante la ejecución del proyecto, como tablas de características, logs, figuras o salidas intermedias.

## Flujo general de trabajo

De forma general, el proyecto sigue esta idea:

1. Partir de los datos originales almacenados en `data/base/`.
2. Reorganizarlos en `data/raw/` para facilitar su manejo.
3. Procesarlos mediante los scripts y módulos del proyecto.

## Notas

Este repositorio está organizado con el objetivo de mantener una separación clara entre:
- los datos originales,
- los datos reorganizados,
- el código fuente,
- y los resultados generados.

Esto facilita la reproducibilidad y el mantenimiento del proyecto.