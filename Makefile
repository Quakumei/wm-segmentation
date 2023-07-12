PYTHON=python3

black:
	$(PYTHON) -m black .

lint: black