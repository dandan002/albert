# Build the paper (needs TeX Live: pdflatex, bibtex, latexmk).
#   make            -> main.pdf
#   make figures    -> regenerate figures/*.pdf from data/research/* (paper venv)
#   make arxiv      -> arxiv.tar.gz with sources, figures and main.bbl (arXiv needs the .bbl)
PY ?= /home/ubuntu/.venvs/paper/bin/python

main.pdf: main.tex macros.tex references.bib bib/*.bib sections/*.tex figures/*.pdf
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

figures:
	cd scripts && $(PY) fig_dynamics.py && $(PY) fig_ladders.py && $(PY) fig_hazard.py

arxiv: main.pdf
	tar czf arxiv.tar.gz main.tex macros.tex main.bbl sections figures

clean:
	latexmk -C main.tex

.PHONY: figures arxiv clean
