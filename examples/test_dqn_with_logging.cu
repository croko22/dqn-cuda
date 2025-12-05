#include "../include/dqn.h"
#include "../include/training_logger.h"
#include <iostream>
#include <vector>
#include <cmath>

// Simple environment for testing
// State: [position, velocity]
// Actions: 0 = left, 1 = nothing, 2 = right
// Goal: reach position 1.0 with velocity close to 0
class SimpleEnvironment
{
private:
    float position_;
    float velocity_;
    int step_count_;
    const int max_steps_ = 200;

public:
    SimpleEnvironment() { reset(); }

    void reset()
    {
        position_ = -0.5f; // Start at left
        velocity_ = 0.0f;
        step_count_ = 0;
    }

    void get_state(float *state)
    {
        state[0] = position_;
        state[1] = velocity_;
    }

    float step(int action, bool &done)
    {
        step_count_++;

        // Apply action: left (-0.1), nothing (0), right (+0.1)
        float acceleration = (action - 1) * 0.1f;
        velocity_ += acceleration;

        // Apply friction
        velocity_ *= 0.95f;

        // Clamp velocity
        if (velocity_ > 0.2f)
            velocity_ = 0.2f;
        if (velocity_ < -0.2f)
            velocity_ = -0.2f;

        // Update position
        position_ += velocity_;

        // Clamp position
        if (position_ > 1.0f)
            position_ = 1.0f;
        if (position_ < -1.0f)
            position_ = -1.0f;

        // Calculate reward
        float distance_to_goal = std::abs(position_ - 1.0f);
        float velocity_penalty = std::abs(velocity_);
        float reward = -distance_to_goal - 0.1f * velocity_penalty;

        // Success bonus
        if (distance_to_goal < 0.1f && velocity_penalty < 0.05f)
        {
            reward += 10.0f;
            done = true;
        }

        // Episode ends after max steps
        if (step_count_ >= max_steps_)
        {
            done = true;
        }

        return reward;
    }

    int get_action_space() const { return 3; }
    int get_state_size() const { return 2; }
};

int main()
{
    std::cout << "DQN Training with Logging Example\n";
    std::cout << "==================================\n\n";

    // Create environment
    SimpleEnvironment env;
    int state_size = env.get_state_size();
    int action_space = env.get_action_space();

    // DQN hyperparameters
    float learning_rate = 0.001f;
    float gamma = 0.99f;
    float epsilon = 1.0f;
    float epsilon_min = 0.01f;
    float epsilon_decay = 0.995f;
    int buffer_capacity = 10000;
    int batch_size = 32;
    int target_update_freq = 10;

    std::cout << "Creating DQN agent...\n";
    std::cout << "  State size: " << state_size << "\n";
    std::cout << "  Action space: " << action_space << "\n";
    std::cout << "  Learning rate: " << learning_rate << "\n";
    std::cout << "  Gamma: " << gamma << "\n";
    std::cout << "  Batch size: " << batch_size << "\n\n";

    DQN agent(state_size, action_space,
              learning_rate, gamma, epsilon, epsilon_min, epsilon_decay,
              buffer_capacity, batch_size, target_update_freq);

    // Create training logger
    TrainingLogger logger("training_log");

    // Training loop
    int num_episodes = 500;
    int global_step = 0;

    std::cout << "Starting training for " << num_episodes << " episodes...\n\n";

    for (int episode = 0; episode < num_episodes; episode++)
    {
        env.reset();
        float state[2];
        env.get_state(state);

        float total_reward = 0.0f;
        int steps = 0;
        bool done = false;
        float episode_loss_sum = 0.0f;
        int train_count = 0;

        while (!done)
        {
            // Select action
            int action = agent.select_action(state, true);

            // Take step in environment
            float reward = env.step(action, done);
            total_reward += reward;
            steps++;
            global_step++;

            float next_state[2];
            env.get_state(next_state);

            // Store transition
            agent.store_experience(state, action, reward, next_state, done);

            // Train agent
            agent.train_step();
            float loss = agent.get_avg_loss();
            if (loss > 0.0f)
            {
                episode_loss_sum += loss;
                train_count++;
            }

            // Update state
            state[0] = next_state[0];
            state[1] = next_state[1];
        }

        // Calculate average loss for episode
        float avg_loss = (train_count > 0) ? episode_loss_sum / train_count : 0.0f;

        // Log episode data
        logger.log_episode(episode, total_reward, steps,
                           agent.get_epsilon(), avg_loss);

        // Print progress every 10 episodes
        if (episode % 10 == 0)
        {
            std::cout << "Episode " << episode
                      << " | Reward: " << total_reward
                      << " | Steps: " << steps
                      << " | Epsilon: " << agent.get_epsilon()
                      << " | Avg Loss: " << avg_loss << "\n";

            // Flush logger periodically
            logger.flush();
        }
    }

    std::cout << "\nTraining complete!\n";
    std::cout << "CSV files saved:\n";
    std::cout << "  - training_log_episodes.csv\n";
    std::cout << "  - training_log_steps.csv\n";
    std::cout << "\nVisualize results with:\n";
    std::cout << "  python visualize_training.py training_log\n";

    return 0;
}
