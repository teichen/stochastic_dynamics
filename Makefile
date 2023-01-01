.PHONY: pip-install clean test

help:
	@echo "pip-install - Install the Python dependencies for the software"
	@echo "clean - clean out the code"
	@echo "test - master testing for the software"

pip-install:
	pip install -r requirements.txt

clean:
	rm -fr build/

test:
	cd tests && pytest kubo_test.py
