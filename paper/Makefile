all:
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex

clean:
	$(RM) *.log
	$(RM) *.aux
	$(RM) *.blg
	$(RM) *.out
	$(RM) *.bbl
