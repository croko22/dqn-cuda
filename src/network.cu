#include "../include/network.h"
#include "../include/optimizer.h"
#include "../include/cuda_utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

// Forward ReLU kernel: f(x) = max(0, x)
__global__ void relu_kernel(float *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// ReLU derivative kernel: f'(x) = 1 if x > 0, else 0
// Multiplies grad by derivative: grad = grad * (hidden > 0)
__global__ void relu_backward_kernel(float *grad, const float *hidden, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        grad[idx] = (hidden[idx] > 0.0f) ? grad[idx] : 0.0f;
    }
}

// Element-wise scale kernel
__global__ void scale_kernel(float *data, float scale, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] *= scale;
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

        // Allocate weights and biases
        cudaMalloc(&d_w1_, w1_size * sizeof(float));
        cudaMalloc(&d_b1_, b1_size * sizeof(float));
        cudaMalloc(&d_w2_, w2_size * sizeof(float));
        cudaMalloc(&d_b2_, b2_size * sizeof(float));

        // Allocate activations
        cudaMalloc(&d_hidden_, hidden_dim_ * sizeof(float));
        cudaMalloc(&d_hidden_pre_relu_, hidden_dim_ * sizeof(float));

        // Allocate gradients
        cudaMalloc(&d_grad_w1_, w1_size * sizeof(float));
        cudaMalloc(&d_grad_b1_, b1_size * sizeof(float));
        cudaMalloc(&d_grad_w2_, w2_size * sizeof(float));
        cudaMalloc(&d_grad_b2_, b2_size * sizeof(float));
        cudaMalloc(&d_grad_hidden_, hidden_dim_ * sizeof(float));

        // Initialize weights with Xavier/He initialization
        float *h_w1 = new float[w1_size];
        float *h_b1 = new float[b1_size];
        float *h_w2 = new float[w2_size];
        float *h_b2 = new float[b2_size];

        std::srand(std::time(nullptr));

        // Xavier initialization for layer 1
        float std_w1 = sqrtf(2.0f / input_dim_);
        for (size_t i = 0; i < w1_size; ++i)
            h_w1[i] = (std::rand() / (float)RAND_MAX - 0.5f) * 2.0f * std_w1;

        for (size_t i = 0; i < b1_size; ++i)
            h_b1[i] = 0.0f;

        // Xavier initialization for layer 2
        float std_w2 = sqrtf(2.0f / hidden_dim_);
        for (size_t i = 0; i < w2_size; ++i)
            h_w2[i] = (std::rand() / (float)RAND_MAX - 0.5f) * 2.0f * std_w2;

        for (size_t i = 0; i < b2_size; ++i)
            h_b2[i] = 0.0f;

        cudaMemcpy(d_w1_, h_w1, w1_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1_, h_b1, b1_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w2_, h_w2, w2_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2_, h_b2, b2_size * sizeof(float), cudaMemcpyHostToDevice);

        // Initialize gradients to zero
        cudaMemset(d_grad_w1_, 0, w1_size * sizeof(float));
        cudaMemset(d_grad_b1_, 0, b1_size * sizeof(float));
        cudaMemset(d_grad_w2_, 0, w2_size * sizeof(float));
        cudaMemset(d_grad_b2_, 0, b2_size * sizeof(float));

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
        cudaFree(d_hidden_pre_relu_);

        cudaFree(d_grad_w1_);
        cudaFree(d_grad_b1_);
        cudaFree(d_grad_w2_);
        cudaFree(d_grad_b2_);
        cudaFree(d_grad_hidden_);
    }

    void forward(const float *input, float *output)
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        // Layer 1: hidden = W1^T * input + b1
        cublasSgemv(handle_, CUBLAS_OP_T, input_dim_, hidden_dim_,
                    &alpha, d_w1_, input_dim_, input, 1, &beta, d_hidden_pre_relu_, 1);

        cublasSaxpy(handle_, hidden_dim_, &alpha, d_b1_, 1, d_hidden_pre_relu_, 1);

        // Copy pre-ReLU activations and apply ReLU
        cudaMemcpy(d_hidden_, d_hidden_pre_relu_, hidden_dim_ * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        apply_relu(d_hidden_, hidden_dim_);

        // Layer 2: output = W2^T * hidden + b2
        cublasSgemv(handle_, CUBLAS_OP_T, hidden_dim_, output_dim_,
                    &alpha, d_w2_, hidden_dim_, d_hidden_, 1, &beta, output, 1);

        cublasSaxpy(handle_, output_dim_, &alpha, d_b2_, 1, output, 1);
    }

    void backward(const float *input, const float *grad_output)
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        // Backprop through layer 2
        // grad_b2 = grad_output
        cudaMemcpy(d_grad_b2_, grad_output, output_dim_ * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        // grad_w2 = hidden * grad_output^T
        // Using: grad_W = outer(hidden, grad_output)
        cublasSger(handle_, hidden_dim_, output_dim_,
                   &alpha, d_hidden_, 1, grad_output, 1,
                   d_grad_w2_, hidden_dim_);

        // grad_hidden = W2 * grad_output
        cublasSgemv(handle_, CUBLAS_OP_N, hidden_dim_, output_dim_,
                    &alpha, d_w2_, hidden_dim_, grad_output, 1,
                    &beta, d_grad_hidden_, 1);

        // Backprop through ReLU
        apply_relu_backward(d_grad_hidden_, d_hidden_pre_relu_, hidden_dim_);

        // Backprop through layer 1
        // grad_b1 = grad_hidden
        cudaMemcpy(d_grad_b1_, d_grad_hidden_, hidden_dim_ * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        // grad_w1 = input * grad_hidden^T
        cublasSger(handle_, input_dim_, hidden_dim_,
                   &alpha, input, 1, d_grad_hidden_, 1,
                   d_grad_w1_, input_dim_);
    }

    void update_weights(Optimizer *optimizer)
    {
        // Update all weights using the optimizer
        int w1_size = input_dim_ * hidden_dim_;
        int w2_size = hidden_dim_ * output_dim_;

        optimizer->step(d_w1_, d_grad_w1_, w1_size);
        optimizer->step(d_b1_, d_grad_b1_, hidden_dim_);
        optimizer->step(d_w2_, d_grad_w2_, w2_size);
        optimizer->step(d_b2_, d_grad_b2_, output_dim_);

        // Zero out gradients after update
        cudaMemset(d_grad_w1_, 0, w1_size * sizeof(float));
        cudaMemset(d_grad_b1_, 0, hidden_dim_ * sizeof(float));
        cudaMemset(d_grad_w2_, 0, w2_size * sizeof(float));
        cudaMemset(d_grad_b2_, 0, output_dim_ * sizeof(float));
    }

    void update_weights_legacy(const float *gradients)
    {
        // Legacy method - simplified update
        float alpha = -0.001f;
        cublasSaxpy(handle_, input_dim_ * hidden_dim_, &alpha, gradients, 1, d_w1_, 1);
    }

    // Getters for weights
    float *get_w1() { return d_w1_; }
    float *get_b1() { return d_b1_; }
    float *get_w2() { return d_w2_; }
    float *get_b2() { return d_b2_; }

    int get_w1_size() const { return input_dim_ * hidden_dim_; }
    int get_b1_size() const { return hidden_dim_; }
    int get_w2_size() const { return hidden_dim_ * output_dim_; }
    int get_b2_size() const { return output_dim_; }

private:
    void apply_relu(float *data, int size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        relu_kernel<<<blocks, threads>>>(data, size);
    }

    void apply_relu_backward(float *grad, const float *hidden_pre_relu, int size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        relu_backward_kernel<<<blocks, threads>>>(grad, hidden_pre_relu, size);
    }

    int input_dim_;
    int output_dim_;
    int hidden_dim_;
    cublasHandle_t handle_;

    // Weights and biases
    float *d_w1_, *d_b1_, *d_w2_, *d_b2_;

    // Activations (for forward pass)
    float *d_hidden_;          // Post-ReLU activations
    float *d_hidden_pre_relu_; // Pre-ReLU activations (for backward)

    // Gradients
    float *d_grad_w1_, *d_grad_b1_, *d_grad_w2_, *d_grad_b2_;
    float *d_grad_hidden_;
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

void Network::backward(const float *input, const float *grad_output)
{
    impl_->backward(input, grad_output);
}

void Network::update_weights(Optimizer *optimizer)
{
    impl_->update_weights(optimizer);
}

void Network::update_weights(const float *gradients)
{
    impl_->update_weights_legacy(gradients);
}

float *Network::get_w1() { return impl_->get_w1(); }
float *Network::get_b1() { return impl_->get_b1(); }
float *Network::get_w2() { return impl_->get_w2(); }
float *Network::get_b2() { return impl_->get_b2(); }

int Network::get_w1_size() const { return impl_->get_w1_size(); }
int Network::get_b1_size() const { return impl_->get_b1_size(); }
int Network::get_w2_size() const { return impl_->get_w2_size(); }
int Network::get_b2_size() const { return impl_->get_b2_size(); }