# Build script for synthetic_branch
EXECUTABLE  := synthetic_branch
# CUDA source files (compiled with cudacc)
CUFILES     := synthetic_branch.cu

include ../../common/common.mk

EXE = synthetic_branch
SRC = synthetic_branch.cu

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(SRC) -o $(EXE) $(NVCC_FLAGS)

clean:
	rm -f $(EXE)
