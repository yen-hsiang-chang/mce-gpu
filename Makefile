NVCC ?= nvcc

BUILD_DIR ?= ./build
SRC_DIRS ?= ./src
EXENAME ?= ./parallel_mce_on_gpus

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s -or -name *.cu)
SRCS_NAMES := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s -or -name *.cu -printf "%f\n")
OBJS := $(SRCS:%=$(BUILD_DIR)/obj/%.o)
EXES := $(SRCS:%=$(BUILD_DIR)/exe/%.exe)
DEPS := $(OBJS:.o=.d)

LDFLAGS := -lgomp
CUDAFLAGS = -w -std=c++11 -O3 -Xcompiler -fopenmp -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80

all: objs exes

objs: $(OBJS)

exes: $(EXES)

$(BUILD_DIR)/exe/%.exe: $(BUILD_DIR)/obj/%.o
	$(MKDIR_P) $(dir $@)
	$(NVCC) $< -o $@ $(LDFLAGS)
	mv $@ $(EXENAME)
	$(RM) -r $(BUILD_DIR)

# cuda source
$(BUILD_DIR)/obj/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

.PHONY: clean

clean:
	$(RM) $(EXENAME)

-include $(DEPS)

MKDIR_P ?= mkdir -p
