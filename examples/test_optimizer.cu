#include "../include/optimizer.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

// Simple test to demonstrate optimizer functionality
void print_array(const char *name, float *d_array, int size)
{
    float *h_array = new float[size];
    cudaMemcpy(h_array, d_array, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << name << ": [";
    for (int i = 0; i < std::min(size, 10); i++)
    {
        std::cout << std::fixed << std::setprecision(4) << h_array[i];
        if (i < std::min(size, 10) - 1)
            std::cout << ", ";
    }
    if (size > 10)
        std::cout << " ...";
    std::cout << "]" << std::endl;

    delete[] h_array;
}

int main()
{
    std::cout << "=== Testing Optimizer ===" << std::endl
              << std::endl;

    const int size = 100;
    const int iterations = 10;

    // Initialize weights and gradients
    float *d_weights_sgd;
    float *d_weights_adam;
    float *d_gradients;

    cudaMalloc(&d_weights_sgd, size * sizeof(float));
    cudaMalloc(&d_weights_adam, size * sizeof(float));
    cudaMalloc(&d_gradients, size * sizeof(float));

    // Initialize with some values
    float *h_weights = new float[size];
    float *h_gradients = new float[size];

    for (int i = 0; i < size; i++)
    {
        h_weights[i] = 1.0f;   // Start with weights = 1.0
        h_gradients[i] = 0.1f; // Constant gradient of 0.1
    }

    cudaMemcpy(d_weights_sgd, h_weights, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_adam, h_weights, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradients, h_gradients, size * sizeof(float), cudaMemcpyHostToDevice);

    // Create optimizers
    Optimizer sgd_optimizer(0.01f, SGD);
    Optimizer adam_optimizer(0.01f, ADAM);

    std::cout << "Initial weights:" << std::endl;
    print_array("Weights", d_weights_sgd, size);
    std::cout << std::endl;

    std::cout << "Gradients:" << std::endl;
    print_array("Gradients", d_gradients, size);
    std::cout << std::endl;

    // Test SGD optimizer
    std::cout << "--- Testing SGD Optimizer ---" << std::endl;
    for (int iter = 0; iter < iterations; iter++)
    {
        sgd_optimizer.step(d_weights_sgd, d_gradients, size);
        std::cout << "Iteration " << iter + 1 << ": ";
        print_array("", d_weights_sgd, size);
    }
    std::cout << std::endl;

    // Reset weights for Adam test
    cudaMemcpy(d_weights_adam, h_weights, size * sizeof(float), cudaMemcpyHostToDevice);

    // Test Adam optimizer
    std::cout << "--- Testing Adam Optimizer ---" << std::endl;
    for (int iter = 0; iter < iterations; iter++)
    {
        adam_optimizer.step(d_weights_adam, d_gradients, size);
        std::cout << "Iteration " << iter + 1 << ": ";
        print_array("", d_weights_adam, size);
    }
    std::cout << std::endl;

    // Cleanup
    delete[] h_weights;
    delete[] h_gradients;
    cudaFree(d_weights_sgd);
    cudaFree(d_weights_adam);
    cudaFree(d_gradients);

    std::cout << "=== Optimizer Test Complete ===" << std::endl;

    return 0;
}
