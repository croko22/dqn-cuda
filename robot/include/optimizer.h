#pragma once

enum OptimizerType
{
    SGD,
    ADAM
};

class Optimizer
{
public:
    Optimizer(float lr, OptimizerType type = ADAM, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    ~Optimizer();

    void step(float *weights, const float *gradients, int size);
    void reset();

private:
    float lr_;
    OptimizerType type_;

    // Adam parameters
    float beta1_;
    float beta2_;
    float epsilon_;
    int t_; // timestep

    // Adam state (momentum and velocity)
    float *d_m_; // first moment
    float *d_v_; // second moment
    int capacity_;

    void step_sgd(float *weights, const float *gradients, int size);
    void step_adam(float *weights, const float *gradients, int size);
};