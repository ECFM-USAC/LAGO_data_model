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
		-t lago-data-model:cpu .

docker-run:
	docker run -d \
		--name lago_container_cpu \
		--user $(shell id -u):$(shell id -g) \
		-p 8888:8888 \
		-v $(shell pwd):/app \
		-v $(shell pwd)/data:/app/data \
		-e PYTHONPATH=/app \
		-e HOME=/tmp \
		lago-data-model:cpu

docker-start: docker-build docker-run
	@echo "CPU container started. Access JupyterLab at http://localhost:8888"

## TensorFlow GPU
docker-build-tf-gpu: banner
	docker build \
		--build-arg BASE_IMAGE=tensorflow/tensorflow:2.17.0rc1-gpu \
		--build-arg INSTALL_TF_DEPS=1 \
		-t lago-data-model-tf:gpu .

docker-run-tf-gpu:
	docker run -d \
		--name lago_container_tf_gpu \
		--user $(shell id -u):$(shell id -g) \
		--gpus all \
		-p 8888:8888 \
		-v $(shell pwd):/app \
		-v $(shell pwd)/data:/app/data \
		-e PYTHONPATH=/app \
		-e HOME=/tmp \
		lago-data-model-tf:gpu

docker-start-tf-gpu: docker-build-tf-gpu docker-run-tf-gpu
	@echo "TensorFlow GPU container started. http://localhost:8888"