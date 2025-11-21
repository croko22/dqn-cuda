#include "../include/network.h"
#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int input_dim = 4;
    int output_dim = 2;

    Network net(input_dim, output_dim);

    float *d_input, *d_output;
    cudaMalloc(&d_input, input_dim * sizeof(float));
    cudaMalloc(&d_output, output_dim * sizeof(float));

    float h_input[] = {1.0f, 0.5f, -0.3f, 0.8f};
    cudaMemcpy(d_input, h_input, input_dim * sizeof(float), cudaMemcpyHostToDevice);

    net.forward(d_input, d_output);

    float h_output[2];
    cudaMemcpy(h_output, d_output, output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Network test:" << std::endl;
    std::cout << "Input: [" << h_input[0] << ", " << h_input[1] << ", "
              << h_input[2] << ", " << h_input[3] << "]" << std::endl;
    std::cout << "Output: [" << h_output[0] << ", " << h_output[1] << "]" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
