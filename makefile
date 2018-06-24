
export FLASK_APP=src/socket/run.py

mysocket:
	flask run

server:
	python3 src/runner.py