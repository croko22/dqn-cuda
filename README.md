# DQN-CUDA: Deep Q-Network Implementation with CUDA

Una implementación completa de Deep Q-Network (DQN) acelerada con CUDA para aprendizaje por refuerzo de alto rendimiento.

## 🚀 Características

- **Aceleración GPU completa** con CUDA
- **Backpropagation completo**: Gradientes exactos calculados en GPU
- **Algoritmo DQN completo** con experience replay y target network
- **Optimizadores avanzados**: SGD y Adam implementados en CUDA
- **Red neuronal flexible** con capas configurables
- **Ejemplos listos para usar** con ambientes de prueba

## 📋 Requisitos

- NVIDIA GPU con capacidad de cómputo 3.5+
- CUDA Toolkit 11.0+
- cuBLAS library
- g++ con soporte C++17
- Make

## 🔧 Compilación y Ejecución

### Compilar el proyecto principal
```bash
make
```

### Ejecutar ejemplos

#### Test del Optimizador
```bash
nvcc -o build/test_optimizer examples/test_optimizer.cu src/optimizer.cu -I./include -lcublas
./build/test_optimizer
```

#### Test del Backward Pass
```bash
nvcc -o build/test_backward examples/test_backward.cu src/network.cu src/optimizer.cu -I./include -lcublas -std=c++17
./build/test_backward
```

#### Test del DQN Completo
```bash
nvcc -o build/test_dqn examples/test_dqn.cu src/dqn.cpp src/network.cu src/optimizer.cu src/replay_buffer.cpp -I./include -lcublas -std=c++17
./build/test_dqn
```

## 📂 Estructura del Proyecto

```
dqn-cuda/
├── include/           # Headers
│   ├── dqn.h         # Agente DQN principal
│   ├── network.h     # Red neuronal
│   ├── optimizer.h   # Optimizadores (SGD, Adam)
│   ├── replay_buffer.h
│   └── cuda_utils.h
├── src/              # Implementaciones
│   ├── dqn.cpp
│   ├── network.cu
│   ├── optimizer.cu
│   ├── replay_buffer.cpp
│   └── main.cu
├── kernels/          # Kernels CUDA
│   ├── activation_kernels.cu
│   ├── linear_kernels.cu
│   └── math_ops.cu
├── examples/         # Ejemplos de uso
│   ├── test_optimizer.cu
│   ├── test_network.cu
│   └── test_dqn.cu
├── docs/             # Documentación
│   ├── dqn_implementation.md
│   └── optimizer_implementation.md
└── Makefile
```

## 🎯 Uso Básico

### Ejemplo: Entrenar un Agente DQN

```cpp
#include "dqn.h"

// Crear agente DQN
DQN agent(state_dim, action_dim);

// Loop de entrenamiento
for (int episode = 0; episode < 1000; episode++) {
    env.reset();
    
    while (!done) {
        // Seleccionar acción (epsilon-greedy)
        int action = agent.select_action(state.data());
        
        // Interactuar con ambiente
        auto [next_state, reward, done] = env.step(action);
        
        // Almacenar experiencia y entrenar
        agent.store_experience(state.data(), action, reward,
                              next_state.data(), done);
        agent.train_step();
    }
    
    // Decaer epsilon
    agent.decay_epsilon();
}
```

## 🧠 Componentes Principales

### 1. DQN Agent (`dqn.h`)
- Política epsilon-greedy para exploración
- Experience replay buffer
- Target network para estabilidad
- Temporal Difference (TD) learning

### 2. Neural Network (`network.h`)
- Red feed-forward con capa oculta
- **Backpropagation completo** con gradientes exactos
- Activación ReLU con derivada
- Forward y backward pass optimizados
- Operaciones matriciales optimizadas con cuBLAS

### 3. Optimizer (`optimizer.h`)
- **SGD**: Stochastic Gradient Descent
- **Adam**: Adaptive Moment Estimation con momentum

### 4. Replay Buffer (`replay_buffer.h`)
- Almacenamiento circular de experiencias
- Muestreo aleatorio uniforme
- Prevención de olvido catastrófico

## 📊 Algoritmo DQN

1. **Inicialización**: Policy network y target network
2. **Exploración**: ε-greedy (ε: 1.0 → 0.01)
3. **Experience Replay**: Guardar (s, a, r, s', done)
4. **Aprendizaje**: 
   - Muestrear batch aleatorio del buffer
   - Calcular TD targets: r + γ max Q_target(s', a')
   - Actualizar policy network minimizando error TD
5. **Actualización**: Copiar pesos a target network periódicamente

## ⚙️ Hiperparámetros

```cpp
learning_rate = 0.001f      // Tasa de aprendizaje (Adam)
gamma = 0.99f               // Factor de descuento
epsilon_start = 1.0f        // Exploración inicial
epsilon_end = 0.01f         // Exploración mínima
epsilon_decay = 0.995f      // Decay por episodio
buffer_capacity = 10000     // Tamaño del replay buffer
batch_size = 32             // Tamaño del batch de entrenamiento
target_update_freq = 100    // Cada cuántos steps actualizar target
```

## 📚 Documentación

- [Documentación Completa del DQN](docs/dqn_implementation.md)
- [Documentación del Optimizador](docs/optimizer_implementation.md)
- [Implementación del Backward Pass](docs/backward_pass_implementation.md)
- [Tutorial: Crear tu propio Ambiente](docs/TUTORIAL.md)

## 📈 Rendimiento

### Ventajas de CUDA:
- ✅ Forward pass en GPU: 10-100x más rápido que CPU
- ✅ Actualización de pesos en paralelo
- ✅ Operaciones matriciales optimizadas (cuBLAS)
- ✅ Sin transferencias CPU-GPU durante entrenamiento

## 🔬 Extensiones Futuras

- [ ] Double DQN
- [ ] Dueling DQN
- [ ] Prioritized Experience Replay
- [ ] N-Step Returns
- [ ] Redes convolucionales (para píxeles)
- [ ] Integración con OpenAI Gym
- [ ] Multi-GPU training

## 📖 Referencias

### Papers Fundamentales
- [Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
- [Human-level control through deep RL](https://www.nature.com/articles/nature14236) (Mnih et al., 2015)
- [Deep RL with Double Q-learning](https://arxiv.org/abs/1509.06461) (van Hasselt et al., 2015)

### Recursos
- [Sutton & Barto: RL Book](http://incompleteideas.net/book/the-book.html)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)

---

**⭐ Si te resulta útil, considera darle una estrella!**
