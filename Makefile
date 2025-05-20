LEPTONICA_PATH = /opt/homebrew/opt/leptonica

CXX = g++
CXXFLAGS = -g -std=c++17 `pkg-config --cflags opencv4 tesseract` -I$(LEPTONICA_PATH)/include
LDFLAGS = `pkg-config --libs opencv4 tesseract` -L$(LEPTONICA_PATH)/lib
EXEC = sudoku_ocr
SRC = sudoku_ocr.cpp
OBJ = sudoku_ocr.o

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)
	rm -rf cells debug
	rm -f warped_sudoku.jpg solved_sudoku.jpg combined_result.jpg

.PHONY: all clean
