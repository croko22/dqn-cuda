CXX = g++
NVCC = nvcc
CXXFLAGS = -Iinclude -O2 -std=c++17
NVCCFLAGS = -Iinclude -O2 -std=c++17
LDFLAGS = -lcublas

CPP_SRC = src/replay_buffer.cpp src/training_logger.cpp
CU_SRC = src/main.cu src/network.cu src/optimizer.cu src/dqn.cu
KERNEL_SRC = kernels/activation_kernels.cu kernels/linear_kernels.cu kernels/math_ops.cu

CPP_OBJ = $(CPP_SRC:src/%.cpp=build/%.o)
CU_OBJ = $(CU_SRC:src/%.cu=build/%.o)
KERNEL_OBJ = $(KERNEL_SRC:kernels/%.cu=build/%.o)
OBJ = $(CPP_OBJ) $(CU_OBJ) $(KERNEL_OBJ)

TARGET = build/dqn_cuda

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

build/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build/%.o: kernels/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f build/*.o build/dqn_cuda
