IDIR = include
CC = cc
CFLAGS = -I$(IDIR)

SRCDIR = src
SRC = main.c insertion_sort.c comparator.c

ODIR=obj
OBJ = $(patsubst %.c, $(ODIR)/%.o, $(SRC))

all: myprogram

$(ODIR)/%.o: $(SRCDIR)/%.c
	@if [ ! -d $(ODIR) ]; then \
		mkdir $(ODIR); \
	fi
	$(CC) -o $@ -c $< $(CFLAGS)

myprogram: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -rf $(ODIR) myprogram
