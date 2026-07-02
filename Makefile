PYTHON := python3

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) demo_continuous_autoregressive.py

clean:
	rm -rf __pycache__
