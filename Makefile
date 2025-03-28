banner: 
	bash banner_ecfm.sh

install: banner
	poetry install

notebook:
	poetry run jupyter lab	

read-data:
	bash scripts/run_data_reader.sh $(file)