all:
	pdflatex dark_energy_corrections.tex
	bibtex dark_energy_corrections
	pdflatex dark_energy_corrections.tex
	pdflatex dark_energy_corrections.tex

clean:
	$(RM) *.log
	$(RM) *.aux
	$(RM) *.blg
	$(RM) *.out
	$(RM) *.bbl
