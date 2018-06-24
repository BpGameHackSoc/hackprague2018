
export FLASK_APP=src/mysocket/run.py

mysocket:
	flask run

server:
	python src/mysocket/run.py