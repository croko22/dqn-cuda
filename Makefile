CXX = g++
NVCC = nvcc
CXXFLAGS = -Iinclude -O2 -std=c++17
NVCCFLAGS = -Iinclude -O2
SRC = src/main.cpp src/dqn.cpp src/replay_buffer.cpp
CU_SRC = src/network.cu src/optimizer.cu \
		  kernels/activation_kernels.cu kernels/linear_kernels.cu kernels/math_ops.cu
OBJ = $(SRC:src/%.cpp=build/%.o) $(CU_SRC:src/%.cu=build/%.o) $(CU_SRC:kernels/%.cu=build/%.o)
TARGET = build/dqn_cuda

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJ)

build/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build/%.o: kernels/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f build/*.o build/dqn_cuda
