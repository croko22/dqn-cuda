#!/bin/bash
# Script de utilidad para DQN-CUDA

set -e

echo "=================================="
echo "   DQN-CUDA Build & Test Script"
echo "=================================="
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_success {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_error {
    echo -e "${RED}✗ $1${NC}"
}

function print_info {
    echo -e "${YELLOW}→ $1${NC}"
}

# Verificar CUDA
if ! command -v nvcc &> /dev/null; then
    print_error "nvcc no encontrado. Por favor instala CUDA Toolkit."
    exit 1
fi
print_success "CUDA Toolkit encontrado: $(nvcc --version | grep release)"

# Menú de opciones
echo ""
echo "Selecciona una opción:"
echo "  1) Compilar proyecto principal (make)"
echo "  2) Compilar y ejecutar test del optimizador"
echo "  3) Compilar y ejecutar test del backward pass"
echo "  4) Compilar y ejecutar test del DQN"
echo "  5) Limpiar build (make clean)"
echo "  6) Compilar todo y ejecutar todos los tests"
echo "  7) Ver documentación"
echo "  0) Salir"
echo ""
read -p "Opción: " option

case $option in
    1)
        print_info "Compilando proyecto principal..."
        make
        print_success "Compilación exitosa!"
        ;;
    2)
        print_info "Compilando test del optimizador..."
        nvcc -o build/test_optimizer examples/test_optimizer.cu src/optimizer.cu -I./include -lcublas
        print_success "Compilación exitosa!"
        print_info "Ejecutando test..."
        ./build/test_optimizer
        ;;
    3)
        print_info "Compilando test del backward pass..."
        nvcc -o build/test_backward examples/test_backward.cu src/network.cu src/optimizer.cu -I./include -lcublas -std=c++17
        print_success "Compilación exitosa!"
        print_info "Ejecutando test..."
        ./build/test_backward
        ;;
    4)
        print_info "Compilando test del DQN..."
        nvcc -o build/test_dqn examples/test_dqn.cu src/dqn.cu src/network.cu src/optimizer.cu src/replay_buffer.cpp -I./include -lcublas -std=c++17
        print_success "Compilación exitosa!"
        print_info "Ejecutando test (esto puede tomar un momento)..."
        ./build/test_dqn
        ;;
    5)
        print_info "Limpiando build..."
        make clean
        rm -f build/test_optimizer build/test_backward build/test_dqn
        print_success "Build limpiado!"
        ;;
    6)
        print_info "Compilando todo..."
        make clean
        make
        
        print_info "Compilando test del optimizador..."
        nvcc -o build/test_optimizer examples/test_optimizer.cu src/optimizer.cu -I./include -lcublas
        
        print_info "Compilando test del backward pass..."
        nvcc -o build/test_backward examples/test_backward.cu src/network.cu src/optimizer.cu -I./include -lcublas -std=c++17
        
        print_info "Compilando test del DQN..."
        nvcc -o build/test_dqn examples/test_dqn.cu src/dqn.cu src/network.cu src/optimizer.cu src/replay_buffer.cpp -I./include -lcublas -std=c++17
        
        print_success "Todo compilado exitosamente!"
        
        echo ""
        print_info "Ejecutando test del optimizador..."
        ./build/test_optimizer
        
        echo ""
        echo "=================================="
        print_info "Ejecutando test del backward pass..."
        ./build/test_backward
        
        echo ""
        echo "=================================="
        print_info "Ejecutando test del DQN..."
        ./build/test_dqn
        
        echo ""
        print_success "¡Todos los tests completados exitosamente!"
        ;;
    7)
        print_info "Documentación disponible:"
        echo ""
        echo "  📄 README.md - Introducción y guía rápida"
        echo "  📄 docs/dqn_implementation.md - Implementación detallada del DQN"
        echo "  📄 docs/optimizer_implementation.md - Documentación del optimizador"
        echo "  📄 docs/PROJECT_SUMMARY.md - Resumen completo del proyecto"
        echo ""
        read -p "¿Abrir README.md? (y/n): " open_readme
        if [ "$open_readme" = "y" ]; then
            less README.md
        fi
        ;;
    0)
        print_info "¡Hasta luego!"
        exit 0
        ;;
    *)
        print_error "Opción inválida"
        exit 1
        ;;
esac

echo ""
print_success "¡Operación completada!"
