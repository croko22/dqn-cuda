#pragma once
#include <string>

class NetworkImpl;
class Optimizer;

class Network
{
public:
    Network(int input_dim, int output_dim);
    ~Network();

    void forward(const float *input, float *output);

    void backward(const float *input, const float *grad_output);

    void update_weights(Optimizer *optimizer);

    void update_weights(const float *gradients);

    void save(const std::string &filename);
    void load(const std::string &filename);

    float *get_w1();
    float *get_b1();
    float *get_w2();
    float *get_b2();

    int get_w1_size() const;
    int get_b1_size() const;
    int get_w2_size() const;
    int get_b2_size() const;

private:
    NetworkImpl *impl_;
};