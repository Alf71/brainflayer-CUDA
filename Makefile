CUDA_PATH ?= /usr/local/cuda-12.8
HOSTCXX ?= g++
CC ?= gcc
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOSTCXX)

BUILD_DIR := build/release
BIN_DIR := bin
TARGET := $(BIN_DIR)/Brainflayer-CUDA

CUDA_SMS ?= 61 75 86 89 120
ARCHFLAGS := $(foreach sm,$(CUDA_SMS),-gencode arch=compute_$(sm),code=sm_$(sm))

INCLUDES := -I. -Isr25519-donna-32bit -Ilib -Ilib/hash -Ilib/V
HOST_FLAGS := -O3 -march=x86-64 -mtune=generic -mssse3 -fPIC -DNDEBUG -pthread
NVCCFLAGS := $(ARCHFLAGS) -std=c++17 -O3 -DNDEBUG -rdc=true --extended-lambda $(INCLUDES) -Xcompiler "$(HOST_FLAGS)"
CXXFLAGS := -std=c++17 $(HOST_FLAGS) $(INCLUDES)
CFLAGS := -std=c17 $(HOST_FLAGS) $(INCLUDES)
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart_static -ldl -lrt -lpthread

SRCS_CU := \
	big_int/BigIntFunc.cu \
	FilterFunc.cu \
	HashFunc.cu \
	IcpFunc.cu \
	Kernels/Kernel.cu \
	Kernels/WorkerBrain.cu \
	Kernels/WorkerBrain_seq.cu \
	Kernels/WorkerMask.cu \
	Kernels/WorkerPRIV.cu \
	Kernels/WorkerPRIV_seq_new.cu \
	Kernels/WorkerPRIV_vanity.cu \
	lib/hash/GPUHash.cu \
	lib/hash/sha3_ver3.cuh \
	lib/secp256k1/GPUConstants.cu \
	lib/secp256k1/secp256k1.cu \
	main.cu \
	SaveFunc.cu \
	sr25519-donna-32bit/ed25519-donna/ed25519.cu \
	sr25519-donna-32bit/sr25519.cu \
	TonFunc.cu

SRCS_CPP := \
	filter.cpp \
	lib/Int.cpp \
	lib/IntGroup.cpp \
	lib/IntMod.cpp \
	lib/Point.cpp \
	lib/SECP256K1.cpp \
	lib/util.cpp \
	lib/V/VBase58.cpp \
	lib/hash/ripemd160.cpp \
	lib/hash/ripemd160_sse.cpp \
	lib/hash/sha256.cpp \
	lib/hash/sha256_sse.cpp \
	sr25519-donna-32bit/dot.cpp

SRCS_C := lib/base58.c

OBJ_CU := $(SRCS_CU:%=$(BUILD_DIR)/%.o)
OBJ_CPP := $(SRCS_CPP:%=$(BUILD_DIR)/%.o)
OBJ_C := $(SRCS_C:%=$(BUILD_DIR)/%.o)
OBJ_ALL := $(OBJ_CU) $(OBJ_CPP) $(OBJ_C)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ_ALL) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cuh.o: %.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(HOSTCXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
