# Project LAGO data model

## 📦 Requisitos (entorno local)

- [Poetry](https://python-poetry.org/docs/#installation)  
- Python 3.11 o superior  

---

### 🚀 Instalación del entorno (local)
```bash
poetry install
```

Para activar el entorno virtual:

```bash
poetry shell
```


### Ejecutar Jupyter Lab
Jupyter lab lo podemos utilizar para ejecutar los notebooks.

```bash
poetry run jupyter lab
```

Esto abrirá una sesión interactiva en tu navegador.

---


## 🐳 Uso con Docker

Puedes ejecutar el entorno completo dentro de un contenedor Docker. Existen dos variantes:

- CPU: entorno ligero, sin soporte para GPU.

- GPU: entorno con PyTorch CUDA y posibilidad de usar torch.cuda (requiere drivers NVIDIA y nvidia-container-toolkit en el host).

### Construir imagen

CPU:
```bash
make docker-build
```

GPU:
```bash
make docker-build-gpu
```

### Ejecutar contenedor

CPU (corre en http://localhost:8888):
```bash

make docker-run
```

GPU (corre en http://localhost:8888):
```bash

make docker-run-gpu
```
Una vez ejecutado el comando puedes entrar al navegador y acceder a jupyter lab


Validación de GPU

Dentro de un notebook, ejecuta:
```python
import torch
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Si aparece tu tarjeta NVIDIA, el contenedor está usando la GPU correctamente.


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