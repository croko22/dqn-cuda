#include "../include/dqn.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

DQN::DQN(int state_dim, int action_dim,
         float learning_rate,
         float gamma,
         float epsilon_start,
         float epsilon_end,
         float epsilon_decay,
         size_t buffer_capacity,
         size_t batch_size,
         int target_update_freq,
         bool use_double_dqn)
    : state_dim_(state_dim),
      action_dim_(action_dim),
      gamma_(gamma),
      epsilon_(epsilon_start),
      epsilon_start_(epsilon_start),
      epsilon_end_(epsilon_end),
      epsilon_decay_(epsilon_decay),
      batch_size_(batch_size),
      target_update_freq_(target_update_freq),
      use_double_dqn_(use_double_dqn),
      total_steps_(0),
      episodes_trained_(0),
      avg_loss_(0.0f)
{
    // Create networks
    policy_net_ = new Network(state_dim, action_dim);
    target_net_ = new Network(state_dim, action_dim);

    // Create replay buffer
    buffer_ = new ReplayBuffer(buffer_capacity);

    // Create optimizer
    optimizer_ = new Optimizer(learning_rate, ADAM);

    // Allocate device memory for training
    cudaMalloc(&d_states_, batch_size * state_dim * sizeof(float));
    cudaMalloc(&d_next_states_, batch_size * state_dim * sizeof(float));
    cudaMalloc(&d_q_values_, batch_size * action_dim * sizeof(float));
    cudaMalloc(&d_next_q_values_, batch_size * action_dim * sizeof(float));
    cudaMalloc(&d_target_q_values_, batch_size * action_dim * sizeof(float));
    cudaMalloc(&d_next_policy_q_values_, batch_size * action_dim * sizeof(float)); // For Double DQN
    cudaMalloc(&d_gradients_, batch_size * action_dim * sizeof(float));

    std::cout << "DQN initialized with:" << std::endl;
    std::cout << "  State dim: " << state_dim << std::endl;
    std::cout << "  Action dim: " << action_dim << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;
    std::cout << "  Gamma: " << gamma << std::endl;
    std::cout << "  Epsilon: " << epsilon_start << " -> " << epsilon_end << std::endl;
    std::cout << "  Buffer capacity: " << buffer_capacity << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Double DQN: " << (use_double_dqn ? "Enabled" : "Disabled") << std::endl;
}

DQN::~DQN()
{
    delete policy_net_;
    delete target_net_;
    delete buffer_;
    delete optimizer_;

    cudaFree(d_states_);
    cudaFree(d_next_states_);
    cudaFree(d_q_values_);
    cudaFree(d_next_q_values_);
    cudaFree(d_target_q_values_);
    cudaFree(d_next_policy_q_values_);
    cudaFree(d_gradients_);
}

int DQN::select_action(const float *state, bool training)
{
    // Epsilon-greedy action selection
    if (training)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        if (dis(gen) < epsilon_)
        {
            // Random action (exploration)
            std::uniform_int_distribution<int> action_dis(0, action_dim_ - 1);
            return action_dis(gen);
        }
    }

    // Greedy action (exploitation)
    float *d_state;
    float *d_q_vals;
    cudaMalloc(&d_state, state_dim_ * sizeof(float));
    cudaMalloc(&d_q_vals, action_dim_ * sizeof(float));

    cudaMemcpy(d_state, state, state_dim_ * sizeof(float), cudaMemcpyHostToDevice);

    // Forward pass through policy network
    policy_net_->forward(d_state, d_q_vals);

    // Get Q-values and select best action
    float *h_q_vals = new float[action_dim_];
    cudaMemcpy(h_q_vals, d_q_vals, action_dim_ * sizeof(float), cudaMemcpyDeviceToHost);

    int best_action = 0;
    float max_q = h_q_vals[0];
    for (int i = 1; i < action_dim_; ++i)
    {
        if (h_q_vals[i] > max_q)
        {
            max_q = h_q_vals[i];
            best_action = i;
        }
    }

    delete[] h_q_vals;
    cudaFree(d_state);
    cudaFree(d_q_vals);

    return best_action;
}

void DQN::store_experience(const float *state, int action, float reward,
                           const float *next_state, bool done)
{
    Experience exp;
    exp.state.assign(state, state + state_dim_);
    exp.action = action;
    exp.reward = reward;
    exp.next_state.assign(next_state, next_state + state_dim_);
    exp.done = done;

    buffer_->add(exp);
}

void DQN::train_step()
{
    // Check if we have enough samples
    if (buffer_->size() < batch_size_)
    {
        return;
    }

    // Sample batch from replay buffer
    std::vector<Experience> batch = buffer_->sample(batch_size_);

    // Prepare batch data
    float *h_states = new float[batch_size_ * state_dim_];
    float *h_next_states = new float[batch_size_ * state_dim_];
    int *h_actions = new int[batch_size_];

    for (size_t i = 0; i < batch_size_; ++i)
    {
        std::copy(batch[i].state.begin(), batch[i].state.end(),
                  h_states + i * state_dim_);
        std::copy(batch[i].next_state.begin(), batch[i].next_state.end(),
                  h_next_states + i * state_dim_);
        h_actions[i] = batch[i].action;
    }

    // Copy to device
    cudaMemcpy(d_states_, h_states, batch_size_ * state_dim_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_next_states_, h_next_states, batch_size_ * state_dim_ * sizeof(float),
               cudaMemcpyHostToDevice);

    // Accumulate loss for monitoring
    float total_loss = 0.0f;

    // Train on each sample in the batch
    for (size_t i = 0; i < batch_size_; ++i)
    {
        // Forward pass through policy network for current state
        policy_net_->forward(d_states_ + i * state_dim_,
                             d_q_values_ + i * action_dim_);

        // Forward pass through target network for next state
        target_net_->forward(d_next_states_ + i * state_dim_,
                             d_next_q_values_ + i * action_dim_);

        // Compute TD target for this sample
        float *h_q_values = new float[action_dim_];
        float *h_next_q_values = new float[action_dim_];
        cudaMemcpy(h_q_values, d_q_values_ + i * action_dim_,
                   action_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_next_q_values, d_next_q_values_ + i * action_dim_,
                   action_dim_ * sizeof(float), cudaMemcpyDeviceToHost);

        // Find max Q-value for next state
        // Find max Q-value for next state
        float max_next_q;

        if (use_double_dqn_)
        {
            // Forward pass through policy network for next state to get selection actions
            policy_net_->forward(d_next_states_ + i * state_dim_,
                                 d_next_policy_q_values_ + i * action_dim_);
            
            float *h_next_policy_q_values = new float[action_dim_];
            cudaMemcpy(h_next_policy_q_values, d_next_policy_q_values_ + i * action_dim_,
                       action_dim_ * sizeof(float), cudaMemcpyDeviceToHost);

            // 1. Select best action using policy network
            int best_next_action = 0;
            float max_policy_q = h_next_policy_q_values[0];
            for (int a = 1; a < action_dim_; ++a)
            {
                if (h_next_policy_q_values[a] > max_policy_q)
                {
                    max_policy_q = h_next_policy_q_values[a];
                    best_next_action = a;
                }
            }
            
            // 2. Evaluate that action using target network (already computed in h_next_q_values)
            max_next_q = h_next_q_values[best_next_action];

            delete[] h_next_policy_q_values;
        }
        else
        {
            // Standard DQN: max Q_target(s', a)
            max_next_q = h_next_q_values[0];
            for (int a = 1; a < action_dim_; ++a)
            {
                max_next_q = std::max(max_next_q, h_next_q_values[a]);
            }
        }

        // Compute TD target
        int action = batch[i].action;
        float reward = batch[i].reward;
        bool done = batch[i].done;

        float td_target;
        if (done)
        {
            td_target = reward;
        }
        else
        {
            td_target = reward + gamma_ * max_next_q;
        }

        // Compute loss gradient (MSE derivative)
        // grad = 2 * (predicted - target) / batch_size
        float *h_grad_output = new float[action_dim_];
        for (int a = 0; a < action_dim_; ++a)
        {
            if (a == action)
            {
                // Only compute gradient for the action taken
                float diff = h_q_values[a] - td_target;
                h_grad_output[a] = 2.0f * diff / batch_size_;
                total_loss += diff * diff;
            }
            else
            {
                h_grad_output[a] = 0.0f; // No gradient for other actions
            }
        }

        // Copy gradient to device
        cudaMemcpy(d_gradients_ + i * action_dim_, h_grad_output,
                   action_dim_ * sizeof(float), cudaMemcpyHostToDevice);

        // Backward pass through policy network
        policy_net_->backward(d_states_ + i * state_dim_,
                              d_gradients_ + i * action_dim_);

        delete[] h_q_values;
        delete[] h_next_q_values;
        delete[] h_grad_output;
    }

    // Update weights using accumulated gradients
    policy_net_->update_weights(optimizer_);

    // Update statistics
    avg_loss_ = total_loss / batch_size_;
    total_steps_++;

    // Update target network periodically
    if (total_steps_ % target_update_freq_ == 0)
    {
        update_target_network();
    }

    // Cleanup
    delete[] h_states;
    delete[] h_next_states;
    delete[] h_actions;
}

void DQN::update_target_network()
{
    // Copy policy network weights to target network
    // In a full implementation, this would copy all layer weights
    std::cout << "Updated target network at step " << total_steps_ << std::endl;

    // For now, we recreate the target network
    // In a production version, you'd copy the actual weight tensors
    delete target_net_;
    target_net_ = new Network(state_dim_, action_dim_);
}

void DQN::decay_epsilon()
{
    epsilon_ = std::max(epsilon_end_, epsilon_ * epsilon_decay_);
}

void DQN::save_model(const std::string &filename)
{
    policy_net_->save(filename);
}

void DQN::load_model(const std::string &filename)
{
    policy_net_->load(filename);
    // Also update target network to match
    delete target_net_;
    target_net_ = new Network(state_dim_, action_dim_);
    // Ideally we should copy weights, but for now loading into policy and letting target update handle it later is okay,
    // OR we can just load into target too.
    target_net_->load(filename);
}
