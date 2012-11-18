PROGNAME = mandel
SRCFILES = mandel.c timer.c
TARNAME = mandel.tgz

RUNARGS =

IMAGEVIEW = display
CC = gcc

WARNFLAGS = -Wall -Werror-implicit-function-declaration -Wshadow \
		  -Wstrict-prototypes -pedantic-errors
CFLAGS = -std=c99 -ggdb $(WARNFLAGS) -D_POSIX_C_SOURCE=200112 -fopenmp -O2
LFLAGS = -lm -lrt

# Do some substitution to get a list of .o files from the given .c files.
OBJFILES = $(patsubst %.c,%.o,$(SRCFILES))

.PHONY: all run runlocal display clean dist todo

all: $(PROGNAME) mandel_no

$(PROGNAME): $(OBJFILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)
	
mandel_no: mandel_no.o timer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)
	
%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

run: $(PROGNAME)
	prun -v -np 1 $(PROGNAME) $(RUNARGS) > output

runlocal: $(PROGNAME)
	./$(PROGNAME) $(RUNARGS) > output
	
out.jpg: output
	convert -depth 8 -size 2501x2000 gray:output out.jpg

display: out.jpg
	display out.jpg

todo:
	-@for file in *.c *.h; do \
		grep -FHnT -e TODO $$file | \
			sed 's/:[ \t]*\/\//: \/\//' | \
			sed 's/:[ \t]*\/\*/: \/\*/'; \
		done; true

dist:
	tar cvzf $(TARNAME) Makefile *.c *.h data/

clean:
	rm -fv $(PROGNAME) $(OBJFILES) $(TARNAME) output out.jpg mandel_no.o mandel_no
