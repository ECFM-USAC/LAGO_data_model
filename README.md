# Project LAGO data model

### 📦 Requisitos

- [Poetry](https://python-poetry.org/docs/#installation)
- Python 3.12 o superior

---

### Instalación del entorno

```bash
poetry install
```

Para activar el entorno virtual:

```bash
poetry shell
```

---

### Ejecutar Jupyter Lab
Jupyter lab lo podemos utilizar para ejecutar los notebooks.

```bash
poetry run jupyter lab
```

Esto abrirá una sesión interactiva en tu navegador.

---

### Procesar un archivo `.lag`

Este proyecto incluye un lector de archivos `.lag` que convierte los datos en un DataFrame y los guarda como archivo `.parquet`.


#### Opción 1: Usar Makefile

```bash
make read-data file=./data/20210518_142318.lag
```

> Esto ejecutará el lector y guardará el archivo como `output/instrument_readings_YYYYMMDDHHMM.parquet`.

#### Opción 2: Ejecutar el script directamente

```bash
poetry run python scripts/run_lag_reader.py ./data/20210518_142318.lag
```

---

### Salida esperada

El archivo generado tendrá este formato:

```
output/instrument_readings_202503291230.parquet
```