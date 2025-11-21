#include "../include/optimizer.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// CUDA kernel for SGD update: weights = weights - lr * gradients
__global__ void sgd_update_kernel(float *weights, const float *gradients, float lr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        weights[idx] -= lr * gradients[idx];
    }
}

// CUDA kernel for Adam optimizer
// Updates first moment (m), second moment (v), and weights
__global__ void adam_update_kernel(
    float *weights,
    const float *gradients,
    float *m,
    float *v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float bias_correction1,
    float bias_correction2,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * gradients[idx];

        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * gradients[idx] * gradients[idx];

        // Compute bias-corrected first moment estimate
        float m_hat = m[idx] / bias_correction1;

        // Compute bias-corrected second raw moment estimate
        float v_hat = v[idx] / bias_correction2;

        // Update weights
        weights[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

Optimizer::Optimizer(float lr, OptimizerType type, float beta1, float beta2, float epsilon)
    : lr_(lr), type_(type), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0),
      d_m_(nullptr), d_v_(nullptr), capacity_(0)
{
}

Optimizer::~Optimizer()
{
    if (d_m_)
        cudaFree(d_m_);
    if (d_v_)
        cudaFree(d_v_);
}

void Optimizer::step(float *weights, const float *gradients, int size)
{
    if (type_ == SGD)
    {
        step_sgd(weights, gradients, size);
    }
    else if (type_ == ADAM)
    {
        step_adam(weights, gradients, size);
    }
}

void Optimizer::step_sgd(float *weights, const float *gradients, int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(weights, gradients, lr_, size);
    cudaDeviceSynchronize();
}

void Optimizer::step_adam(float *weights, const float *gradients, int size)
{
    // Allocate Adam state buffers if needed
    if (capacity_ < size)
    {
        if (d_m_)
            cudaFree(d_m_);
        if (d_v_)
            cudaFree(d_v_);

        cudaMalloc(&d_m_, size * sizeof(float));
        cudaMalloc(&d_v_, size * sizeof(float));
        cudaMemset(d_m_, 0, size * sizeof(float));
        cudaMemset(d_v_, 0, size * sizeof(float));
        capacity_ = size;
    }

    // Increment timestep
    t_++;

    // Compute bias correction terms
    float bias_correction1 = 1.0f - powf(beta1_, t_);
    float bias_correction2 = 1.0f - powf(beta2_, t_);

    // Launch kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    adam_update_kernel<<<blocks, threads>>>(
        weights, gradients, d_m_, d_v_,
        lr_, beta1_, beta2_, epsilon_,
        bias_correction1, bias_correction2,
        size);
    cudaDeviceSynchronize();
}

void Optimizer::reset()
{
    t_ = 0;
    if (d_m_)
        cudaMemset(d_m_, 0, capacity_ * sizeof(float));
    if (d_v_)
        cudaMemset(d_v_, 0, capacity_ * sizeof(float));
}