#pragma once

class NetworkImpl;

class Network
{
public:
    Network(int input_dim, int output_dim);
    ~Network();
    void forward(const float *input, float *output);
    void update_weights(const float *gradients);

private:
    NetworkImpl *impl_;
};