BIB = bibtex
TEX = pdflatex

FILE = indexing_paper

ADDITIONAL = social/social

$(FILE).pdf: $(FILE).bbl
	$(TEX) $(FILE)
	$(TEX) $(FILE)

$(FILE).bbl: $(FILE).bib $(FILE).aux
	$(BIB) $(FILE).aux
	
$(FILE).aux:
	$(TEX) $(FILE)
	
.PHONY: clean

clean:
	rm -f $(FILE).pdf *.aux *.bbl *.blg *.out *.log
	rm -f $(ADDITIONAL).tex.aux