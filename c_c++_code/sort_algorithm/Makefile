IDIR = include
CC = cc
CFLAGS = -I$(IDIR)

SRCDIR = src
SRC = main.c insertion_sort.c comparator.c

ODIR=obj
OBJ = $(patsubst %.c, $(ODIR)/%.o, $(SRC))

EXECUTABLE = sorting_algorithm
all: $(EXECUTABLE)

$(ODIR)/%.o: $(SRCDIR)/%.c
	@if [ ! -d $(ODIR) ]; then \
		mkdir $(ODIR); \
	fi
	$(CC) -o $@ -c $< $(CFLAGS)

$(EXECUTABLE): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -rf $(ODIR) $(EXECUTABLE)
