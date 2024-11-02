make_venv:
	if [ ! -d "$(CURDIR)/venv" ]; then python -m venv $(CURDIR)/venv; fi

check_env: make_venv
ifndef VIRTUAL_ENV
	$(error Not in a python venv. Run `source venv/bin/activate`)
endif

update_env: check_env
	pip install --quiet --editable .

generate_graphs: update_env
	python src/redshift/scratch.py

check_pdflatex:
ifeq (, $(shell which pdflatex))
  $(error "No pdflatex in $(PATH). Consider executing: sudo apt-get install texlive-latex-extra")
endif

paper: generate_graphs check_pdflatex
	make -C paper
