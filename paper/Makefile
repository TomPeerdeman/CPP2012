BIB = bibtex
TEX = pdflatex

FILE = indexing_paper

ADDITIONAL = social/social

$(FILE).pdf: $(FILE).bbl
	$(TEX) $(FILE) > out
	rm -f out
	$(TEX) $(FILE)

$(FILE).bbl: $(FILE).bib $(FILE).aux
	$(BIB) $(FILE).aux
	
$(FILE).aux:
	$(TEX) $(FILE) > out
	rm -f out
	
.PHONY: clean bib pdf cleanbib

clean:
	rm -f $(FILE).pdf *.aux *.bbl *.blg *.out *.log
	rm -f $(ADDITIONAL).tex.aux
	
bib: $(FILE).bbl
	
pdf: $(FILE).bbl
	$(TEX) $(FILE) > out
	rm -f out
	$(TEX) $(FILE)
	
cleanbib:
	rm -f *.bbl *.blg