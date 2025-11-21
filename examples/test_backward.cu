#include "../include/network.h"
#include "../include/optimizer.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

// Helper function to print array
void print_array(const char *name, const float *d_array, int size, int max_print = 5)
{
    float *h_array = new float[size];
    cudaMemcpy(h_array, d_array, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << name << ": [";
    for (int i = 0; i < std::min(size, max_print); i++)
    {
        std::cout << std::fixed << std::setprecision(4) << h_array[i];
        if (i < std::min(size, max_print) - 1)
            std::cout << ", ";
    }
    if (size > max_print)
        std::cout << " ...";
    std::cout << "]" << std::endl;

    delete[] h_array;
}

// Compute MSE loss gradient
__global__ void mse_grad_kernel(float *grad_output, const float *output,
                                const float *target, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        grad_output[idx] = 2.0f * (output[idx] - target[idx]) / size;
    }
}

int main()
{
    std::cout << "=== Testing Network Backward Pass ===" << std::endl
              << std::endl;

    const int input_dim = 4;
    const int output_dim = 2;
    const int num_samples = 10;
    const int iterations = 100;

    // Create network and optimizer
    Network net(input_dim, output_dim);
    Optimizer optimizer(0.01f, ADAM); // Learning rate 0.01

    std::cout << "Network: " << input_dim << " -> 128 -> " << output_dim << std::endl;
    std::cout << "Optimizer: Adam (lr=0.01)" << std::endl
              << std::endl;

    // Allocate device memory
    float *d_input;
    float *d_output;
    float *d_target;
    float *d_grad_output;

    cudaMalloc(&d_input, input_dim * sizeof(float));
    cudaMalloc(&d_output, output_dim * sizeof(float));
    cudaMalloc(&d_target, output_dim * sizeof(float));
    cudaMalloc(&d_grad_output, output_dim * sizeof(float));

    // Create a simple training dataset
    // Target: output[0] = sum(input), output[1] = product(input[0:2])
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;

    for (int i = 0; i < num_samples; i++)
    {
        std::vector<float> input(input_dim);
        for (int j = 0; j < input_dim; j++)
        {
            input[j] = (rand() % 100) / 100.0f; // Random [0, 1]
        }

        std::vector<float> target(output_dim);
        target[0] = 0.0f;
        for (int j = 0; j < input_dim; j++)
        {
            target[0] += input[j];
        }
        target[0] /= input_dim;          // Average
        target[1] = input[0] * input[1]; // Product of first two

        inputs.push_back(input);
        targets.push_back(target);
    }

    std::cout << "Training dataset: " << num_samples << " samples" << std::endl;
    std::cout << "Task: Learn f([x1,x2,x3,x4]) = [avg(x), x1*x2]" << std::endl
              << std::endl;

    // Training loop
    std::cout << "Starting training..." << std::endl;

    for (int iter = 0; iter < iterations; iter++)
    {
        float total_loss = 0.0f;

        // Train on all samples
        for (int sample = 0; sample < num_samples; sample++)
        {
            // Copy input and target to device
            cudaMemcpy(d_input, inputs[sample].data(),
                       input_dim * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_target, targets[sample].data(),
                       output_dim * sizeof(float), cudaMemcpyHostToDevice);

            // Forward pass
            net.forward(d_input, d_output);

            // Compute loss gradient (MSE)
            int threads = 256;
            int blocks = (output_dim + threads - 1) / threads;
            mse_grad_kernel<<<blocks, threads>>>(d_grad_output, d_output, d_target, output_dim);
            cudaDeviceSynchronize();

            // Compute loss for monitoring
            float *h_output = new float[output_dim];
            cudaMemcpy(h_output, d_output, output_dim * sizeof(float), cudaMemcpyDeviceToHost);
            for (int j = 0; j < output_dim; j++)
            {
                float diff = h_output[j] - targets[sample][j];
                total_loss += diff * diff;
            }
            delete[] h_output;

            // Backward pass
            net.backward(d_input, d_grad_output);

            // Update weights
            net.update_weights(&optimizer);
        }

        total_loss /= num_samples;

        // Print progress
        if (iter % 10 == 0 || iter == iterations - 1)
        {
            std::cout << "Iteration " << std::setw(3) << iter + 1
                      << " - Loss: " << std::fixed << std::setprecision(6) << total_loss
                      << std::endl;
        }
    }

    std::cout << std::endl
              << "=== Testing Learned Function ===" << std::endl
              << std::endl;

    // Test on training samples
    for (int i = 0; i < std::min(5, num_samples); i++)
    {
        cudaMemcpy(d_input, inputs[i].data(),
                   input_dim * sizeof(float), cudaMemcpyHostToDevice);

        net.forward(d_input, d_output);

        float *h_input = inputs[i].data();
        float *h_output = new float[output_dim];
        float *h_target = targets[i].data();
        cudaMemcpy(h_output, d_output, output_dim * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Sample " << i + 1 << ":" << std::endl;
        std::cout << "  Input:  [";
        for (int j = 0; j < input_dim; j++)
        {
            std::cout << std::fixed << std::setprecision(2) << h_input[j];
            if (j < input_dim - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  Target: [";
        for (int j = 0; j < output_dim; j++)
        {
            std::cout << std::fixed << std::setprecision(4) << h_target[j];
            if (j < output_dim - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  Output: [";
        for (int j = 0; j < output_dim; j++)
        {
            std::cout << std::fixed << std::setprecision(4) << h_output[j];
            if (j < output_dim - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  Error:  [";
        for (int j = 0; j < output_dim; j++)
        {
            float error = std::abs(h_output[j] - h_target[j]);
            std::cout << std::fixed << std::setprecision(4) << error;
            if (j < output_dim - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl
                  << std::endl;

        delete[] h_output;
    }

    // Test on new samples
    std::cout << "=== Testing on New Samples ===" << std::endl
              << std::endl;

    for (int i = 0; i < 3; i++)
    {
        float h_input[input_dim];
        for (int j = 0; j < input_dim; j++)
        {
            h_input[j] = (rand() % 100) / 100.0f;
        }

        float expected_avg = 0.0f;
        for (int j = 0; j < input_dim; j++)
        {
            expected_avg += h_input[j];
        }
        expected_avg /= input_dim;
        float expected_prod = h_input[0] * h_input[1];

        cudaMemcpy(d_input, h_input, input_dim * sizeof(float), cudaMemcpyHostToDevice);
        net.forward(d_input, d_output);

        float *h_output = new float[output_dim];
        cudaMemcpy(h_output, d_output, output_dim * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "New Sample " << i + 1 << ":" << std::endl;
        std::cout << "  Input:    [";
        for (int j = 0; j < input_dim; j++)
        {
            std::cout << std::fixed << std::setprecision(2) << h_input[j];
            if (j < input_dim - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  Expected: [" << std::fixed << std::setprecision(4)
                  << expected_avg << ", " << expected_prod << "]" << std::endl;
        std::cout << "  Predicted: [" << std::fixed << std::setprecision(4)
                  << h_output[0] << ", " << h_output[1] << "]" << std::endl
                  << std::endl;

        delete[] h_output;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_target);
    cudaFree(d_grad_output);

    std::cout << "=== Test Complete ===" << std::endl;

    return 0;
}
