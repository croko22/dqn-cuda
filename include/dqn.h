#pragma once

#include "network.h"
#include "replay_buffer.h"
#include "optimizer.h"

class DQN
{
public:
    DQN(int state_dim, int action_dim,
        float learning_rate = 0.001f,
        float gamma = 0.99f,
        float epsilon_start = 1.0f,
        float epsilon_end = 0.01f,
        float epsilon_decay = 0.995f,
        size_t buffer_capacity = 10000,
        size_t batch_size = 64,
        int target_update_freq = 10,
        bool use_double_dqn = false);

    ~DQN();

    int select_action(const float *state, bool training = true);

    void store_experience(const float *state, int action, float reward,
                          const float *next_state, bool done);

    void train_step();

    void update_target_network();

    float get_epsilon() const { return epsilon_; }
    void set_epsilon(float epsilon) { epsilon_ = epsilon; }
    void decay_epsilon();

    int get_steps() const { return total_steps_; }
    float get_avg_loss() const { return avg_loss_; }
    float get_last_q_value() const { return last_q_value_; }

    void save_model(const std::string &filename);
    void load_model(const std::string &filename);

private:
    Network *policy_net_;
    Network *target_net_;

    ReplayBuffer *buffer_;
    Optimizer *optimizer_;

    int state_dim_;
    int action_dim_;
    float gamma_;
    float epsilon_;
    float epsilon_start_;
    float epsilon_end_;
    float epsilon_decay_;
    size_t batch_size_;
    int target_update_freq_;
    bool use_double_dqn_;

    int total_steps_;
    int episodes_trained_;
    float avg_loss_;
    float last_q_value_;

    float *d_states_;
    float *d_next_states_;
    float *d_q_values_;
    float *d_next_q_values_;
    float *d_target_q_values_;
    float *d_next_policy_q_values_; // For Double DQN
    float *d_gradients_;
};