#include "../include/network.h"
#include "../include/cuda_utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

__global__ void relu_kernel(float *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

class NetworkImpl
{
public:
    NetworkImpl(int input_dim, int output_dim)
        : input_dim_(input_dim), output_dim_(output_dim),
          hidden_dim_(128)
    {
        cublasCreate(&handle_);

        size_t w1_size = input_dim_ * hidden_dim_;
        size_t b1_size = hidden_dim_;
        size_t w2_size = hidden_dim_ * output_dim_;
        size_t b2_size = output_dim_;

        cudaMalloc(&d_w1_, w1_size * sizeof(float));
        cudaMalloc(&d_b1_, b1_size * sizeof(float));
        cudaMalloc(&d_w2_, w2_size * sizeof(float));
        cudaMalloc(&d_b2_, b2_size * sizeof(float));
        cudaMalloc(&d_hidden_, hidden_dim_ * sizeof(float));

        float *h_w1 = new float[w1_size];
        float *h_b1 = new float[b1_size];
        float *h_w2 = new float[w2_size];
        float *h_b2 = new float[b2_size];

        std::srand(std::time(nullptr));
        for (size_t i = 0; i < w1_size; ++i)
            h_w1[i] = (std::rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        for (size_t i = 0; i < b1_size; ++i)
            h_b1[i] = 0.0f;
        for (size_t i = 0; i < w2_size; ++i)
            h_w2[i] = (std::rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        for (size_t i = 0; i < b2_size; ++i)
            h_b2[i] = 0.0f;

        cudaMemcpy(d_w1_, h_w1, w1_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1_, h_b1, b1_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w2_, h_w2, w2_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2_, h_b2, b2_size * sizeof(float), cudaMemcpyHostToDevice);

        delete[] h_w1;
        delete[] h_b1;
        delete[] h_w2;
        delete[] h_b2;
    }

    ~NetworkImpl()
    {
        cublasDestroy(handle_);
        cudaFree(d_w1_);
        cudaFree(d_b1_);
        cudaFree(d_w2_);
        cudaFree(d_b2_);
        cudaFree(d_hidden_);
    }

    void forward(const float *input, float *output)
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        cublasSgemv(handle_, CUBLAS_OP_T, input_dim_, hidden_dim_,
                    &alpha, d_w1_, input_dim_, input, 1, &beta, d_hidden_, 1);

        cublasSaxpy(handle_, hidden_dim_, &alpha, d_b1_, 1, d_hidden_, 1);

        apply_relu(d_hidden_, hidden_dim_);

        cublasSgemv(handle_, CUBLAS_OP_T, hidden_dim_, output_dim_,
                    &alpha, d_w2_, hidden_dim_, d_hidden_, 1, &beta, output, 1);

        cublasSaxpy(handle_, output_dim_, &alpha, d_b2_, 1, output, 1);
    }

    void update_weights(const float *gradients)
    {
        float alpha = -0.001f;
        cublasSaxpy(handle_, input_dim_ * hidden_dim_, &alpha, gradients, 1, d_w1_, 1);
    }

private:
    void apply_relu(float *data, int size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        relu_kernel<<<blocks, threads>>>(data, size);
    }

    int input_dim_;
    int output_dim_;
    int hidden_dim_;
    cublasHandle_t handle_;
    float *d_w1_, *d_b1_, *d_w2_, *d_b2_;
    float *d_hidden_;
};

Network::Network(int input_dim, int output_dim)
{
    impl_ = new NetworkImpl(input_dim, output_dim);
}

Network::~Network()
{
    delete impl_;
}

void Network::forward(const float *input, float *output)
{
    impl_->forward(input, output);
}

void Network::update_weights(const float *gradients)
{
    impl_->update_weights(gradients);
}