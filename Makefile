IDIR = dep
ODIR  = bin
SDIR  = src
CC    = g++ 
FLAGS = -g -I $(IDIR) 

main.exe: $(ODIR)/driver.o $(ODIR)/Matrix.o $(ODIR)/MLToolKit.o
	$(CC) $(FLAGS) $^ -o $(ODIR)/$@

$(ODIR)/driver.o: $(SDIR)/driver.cpp $(ODIR)/Matrix.o $(ODIR)/MLToolKit.o
	$(CC) $(FLAGS) -c $< -o $@

$(ODIR)/MLToolKit.o: $(IDIR)/MLToolKit.cpp $(IDIR)/MLToolKit.hpp $(ODIR)/Matrix.o
	$(CC) $(FLAGS) -c $< -o $@

$(ODIR)/Matrix.o: $(IDIR)/Matrix.cpp $(IDIR)/Matrix.hpp
	$(CC) $(FLAGS) -c $< -o $@

clean:
	rm bin/main.exe bin/*.o

