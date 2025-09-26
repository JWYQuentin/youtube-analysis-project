.PHONY: init train app clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit

init:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt streamlit joblib

train: init
	$(PYTHON) refined_model.py

app: init
	$(STREAMLIT) run app.py

clean:
	rm -rf $(VENV) __pycache__ */__pycache__ .pytest_cache .streamlit
