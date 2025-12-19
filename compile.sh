#!/bin/bash

echo "Compilando train_navigator.cu..."

# Flags de compilación
NVCC_FLAGS="-std=c++17 -O2 -I./include -I./src"
LIBS="-lcuda -lcudart -lpthread -lboost_system"

# Archivos a compilar
CUDA_FILES="src/train_navigator.cu src/dqn.cu src/network.cu src/optimizer.cu"
CPP_FILES="src/navigator_enviroment.cpp src/ev3_interface.cpp src/replay_buffer.cpp src/training_logger.cpp"

# Compilar
nvcc $NVCC_FLAGS -o train_navigator $CUDA_FILES $CPP_FILES $LIBS

if [ $? -eq 0 ]; then
    echo "Compilación exitosa!"
    echo "Ejecutable creado: train_navigator"
else
    echo "Error en la compilación"
    exit 1
fi