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

## Docker (GPU - PyTorch)
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
	@echo "GPU container started. Access JupyterLab at http://localhost:8888"

# Docker (TensorFlow CPU)
docker-build-tf:
	docker build \
		--build-arg BASE_IMAGE=python:3.12-slim \
		--build-arg INSTALL_GPU_DEPS=0 \
		-t lago-data-model-tf:cpu .

docker-run-tf:
	docker run -d \
		--name lago_container_tf_cpu \
		-p 8888:8888 \
		-v $(shell pwd):/app \
		-v $(shell pwd)/data:/app/data \
		-e PYTHONPATH=/app \
		lago-data-model-tf:cpu

docker-start-tf: docker-build-tf docker-run-tf
	@echo "TensorFlow CPU container started. Access JupyterLab at http://localhost:8888"


## Docker (TensorFlow GPU)
docker-build-tf-gpu:
	docker build \
		--build-arg BASE_IMAGE=python:3.12-slim \
		--build-arg INSTALL_GPU_DEPS=1 \
		-t lago-data-model-tf:gpu .

docker-run-tf-gpu:
	docker run -d \
		--name lago_container_tf_gpu \
		--gpus all \
		-p 8888:8888 \
		-v $(shell pwd):/app \
		-v $(shell pwd)/data:/app/data \
		-e PYTHONPATH=/app \
		lago-data-model-tf:gpu

docker-start-tf-gpu: docker-build-tf-gpu docker-run-tf-gpu
	@echo "TensorFlow GPU container started. Access JupyterLab at http://localhost:8888"