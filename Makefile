IDIR = dep
ODIR  = bin
SDIR  = src
CC    = g++ 
FLAGS = -g -I $(IDIR) 

main.exe: $(ODIR)/driver.o $(ODIR)/MatrixMath.o $(ODIR)/MLToolKit.o
	$(CC) $(FLAGS) $^ -o $(ODIR)/$@

$(ODIR)/driver.o: $(SDIR)/driver.cpp $(ODIR)/MatrixMath.o $(ODIR)/MLToolKit.o
	$(CC) $(FLAGS) -c $< -o $@

$(ODIR)/MLToolKit.o: $(IDIR)/MLToolKit.cpp $(IDIR)/MLToolKit.hpp $(ODIR)/MatrixMath.o
	$(CC) $(FLAGS) -c $< -o $@

$(ODIR)/MatrixMath.o: $(IDIR)/MatrixMath.cpp $(IDIR)/MatrixMath.hpp
	$(CC) $(FLAGS) -c $< -o $@

clean:
	rm bin/main.exe bin/*.o

