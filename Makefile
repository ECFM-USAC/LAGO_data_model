banner: 
	bash banner_ecfm.sh

install: banner
	poetry install

notebook: banner
	poetry run jupyter lab	

read-data: banner
	bash scripts/run_data_reader.sh $(file)


## Docker
docker-build: banner
	docker build -t lago-data-model .

docker-run: 
	docker run -d \
		--name lago_container \
		-p 8888:8888 \
		-v $(shell pwd):/app \
		-v $(shell pwd)/data:/app/data \
		-e PYTHONPATH=/app \
		lago-data-model

docker-start: docker-build docker-run
	@echo "Container started. Access JupyterLab at http://192.168.30.70:8888"