#pragma once

#include "network.h"
#include "replay_buffer.h"
#include "optimizer.h"

class DQN
{
public:
    // Constructor with hyperparameters
    DQN(int state_dim, int action_dim,
        float learning_rate = 0.001f,
        float gamma = 0.99f,
        float epsilon_start = 1.0f,
        float epsilon_end = 0.01f,
        float epsilon_decay = 0.995f,
        size_t buffer_capacity = 10000,
        size_t batch_size = 64,
        int target_update_freq = 10);

    ~DQN();

    // Select action using epsilon-greedy policy
    int select_action(const float *state, bool training = true);

    // Store experience in replay buffer
    void store_experience(const float *state, int action, float reward,
                          const float *next_state, bool done);

    // Train on a batch from replay buffer
    void train_step();

    // Update target network with policy network weights
    void update_target_network();

    // Get/set epsilon for exploration
    float get_epsilon() const { return epsilon_; }
    void set_epsilon(float epsilon) { epsilon_ = epsilon; }
    void decay_epsilon();

    // Get training statistics
    int get_steps() const { return total_steps_; }
    float get_avg_loss() const { return avg_loss_; }

private:
    // Networks
    Network *policy_net_;
    Network *target_net_;

    // Components
    ReplayBuffer *buffer_;
    Optimizer *optimizer_;

    // Hyperparameters
    int state_dim_;
    int action_dim_;
    float gamma_;   // Discount factor
    float epsilon_; // Current exploration rate
    float epsilon_start_;
    float epsilon_end_;
    float epsilon_decay_;
    size_t batch_size_;
    int target_update_freq_;

    // Training state
    int total_steps_;
    int episodes_trained_;
    float avg_loss_;

    // Device memory for training
    float *d_states_;
    float *d_next_states_;
    float *d_q_values_;
    float *d_next_q_values_;
    float *d_target_q_values_;
    float *d_gradients_;

    // Helper functions
    void compute_td_targets(const std::vector<Experience> &batch,
                            float *h_target_q_values);
    void compute_gradients(const float *q_values, const float *targets,
                           const int *actions, int batch_size);
};