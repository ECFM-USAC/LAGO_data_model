banner: 
	bash banner_ecfm.sh

install: banner
	poetry install

notebook: banner
	poetry run jupyter lab	

read-data: banner
	bash scripts/run_data_reader.sh $(file)

## Docker (CPU por defecto)
docker-build: banner
	docker build \
		--build-arg BASE_IMAGE=python:3.12-slim \
		--build-arg INSTALL_GPU_DEPS=0 \
		-t lago-data-model:cpu .

docker-run: 
	docker run -d \
		--name lago_container_cpu \
		-p 8888:8888 \
		-v $(shell pwd):/app \
		-v $(shell pwd)/data:/app/data \
		-e PYTHONPATH=/app \
		lago-data-model:cpu

docker-start: docker-build docker-run
	@echo "CPU container started. Access JupyterLab at http://localhost:8888"

## Docker (GPU)
docker-build-gpu: banner
	docker build \
		--build-arg BASE_IMAGE=pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime \
		--build-arg INSTALL_GPU_DEPS=1 \
		-t lago-data-model:gpu .

docker-run-gpu:
	docker run -d \
		--name lago_container_gpu \
		--gpus all \
		-p 8888:8888 \
		-v $(shell pwd):/app \
		-v $(shell pwd)/data:/app/data \
		-e PYTHONPATH=/app \
		lago-data-model:gpu

docker-start-gpu: docker-build-gpu docker-run-gpu
	@echo "GPU container started. Access JupyterLab at http://localhost:8889"
