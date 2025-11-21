# DQN-CUDA: Deep Q-Network Implementation with CUDA

Una implementación completa de Deep Q-Network (DQN) acelerada con CUDA para aprendizaje por refuerzo de alto rendimiento.

## Características

* Aceleración GPU completa con CUDA
* Backpropagation completo con gradientes exactos en GPU
* Algoritmo DQN con experience replay y target network
* Optimizadores implementados en CUDA (SGD y Adam)
* Red neuronal configurable
* Ejemplos listos para usar

## Requisitos

* NVIDIA GPU con capacidad de cómputo 3.5+
* CUDA Toolkit 11.0+
* cuBLAS
* g++ con C++17
* Make

## Compilación y Ejecución

### Compilar el proyecto principal

```bash
make
```

### Test del Optimizador

```bash
nvcc -o build/test_optimizer examples/test_optimizer.cu src/optimizer.cu -I./include -lcublas
./build/test_optimizer
```

### Test del Backward Pass

```bash
nvcc -o build/test_backward examples/test_backward.cu src/network.cu src/optimizer.cu -I./include -lcublas -std=c++17
./build/test_backward
```

### Test del DQN Completo

```bash
nvcc -o build/test_dqn examples/test_dqn.cu src/dqn.cpp src/network.cu src/optimizer.cu src/replay_buffer.cpp -I./include -lcublas -std=c++17
./build/test_dqn
```

## Estructura del Proyecto

```
dqn-cuda/
├── include/
│   ├── dqn.h
│   ├── network.h
│   ├── optimizer.h
│   ├── replay_buffer.h
│   └── cuda_utils.h
├── src/
│   ├── dqn.cpp
│   ├── network.cu
│   ├── optimizer.cu
│   ├── replay_buffer.cpp
│   └── main.cu
├── kernels/
│   ├── activation_kernels.cu
│   ├── linear_kernels.cu
│   └── math_ops.cu
├── examples/
│   ├── test_optimizer.cu
│   ├── test_network.cu
│   └── test_dqn.cu
├── docs/
│   ├── dqn_implementation.md
│   └── optimizer_implementation.md
└── Makefile
```

## Uso Básico

```cpp
#include "dqn.h"

DQN agent(state_dim, action_dim);

for (int episode = 0; episode < 1000; episode++) {
    env.reset();
    while (!done) {
        int action = agent.select_action(state.data());
        auto [next_state, reward, done] = env.step(action);

        agent.store_experience(state.data(), action, reward,
                               next_state.data(), done);
        agent.train_step();
    }
    agent.decay_epsilon();
}
```

## Componentes Principales

### DQN Agent

* Exploración epsilon-greedy
* Replay buffer
* Target network
* Temporal Difference learning

### Neural Network

* Red feed-forward
* Backpropagation completo
* ReLU y derivada
* cuBLAS para operaciones matriciales

### Optimizer

* SGD
* Adam

### Replay Buffer

* Memoria circular
* Muestreo aleatorio
* Evita olvido catastrófico

## Algoritmo DQN

1. Inicializar redes
2. Seleccionar acción con epsilon-greedy
3. Guardar transiciones
4. Muestrear batch del buffer
5. Calcular TD target
6. Actualizar la policy network
7. Actualizar la target network periódicamente

## Hiperparámetros

```cpp
learning_rate = 0.001f
gamma = 0.99f
epsilon_start = 1.0f
epsilon_end = 0.01f
epsilon_decay = 0.995f
buffer_capacity = 10000
batch_size = 32
target_update_freq = 100
```

## Rendimiento

* Forward pass 10–100x más rápido que CPU
* Actualización de pesos paralelizada
* Operaciones matriciales optimizadas
* Sin transferencias innecesarias CPU-GPU

## Referencias
- [Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
- [Human-level control through deep RL](https://www.nature.com/articles/nature14236) (Mnih et al., 2015)
- [Deep RL with Double Q-learning](https://arxiv.org/abs/1509.06461) (van Hasselt et al., 2015)

### Recursos
- [Sutton & Barto: RL Book](http://incompleteideas.net/book/the-book.html)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)