#include "../include/dqn.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Simple CartPole-like environment simulation
class SimpleEnvironment
{
public:
    SimpleEnvironment(int state_dim, int action_dim)
        : state_dim_(state_dim), action_dim_(action_dim), step_count_(0)
    {
        reset();
    }

    void reset()
    {
        state_.resize(state_dim_);
        // Initialize state randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < state_dim_; ++i)
        {
            state_[i] = dis(gen);
        }
        step_count_ = 0;
    }

    // Returns: next_state, reward, done
    std::tuple<std::vector<float>, float, bool> step(int action)
    {
        step_count_++;

        // Simple dynamics: state changes based on action
        std::vector<float> next_state = state_;
        for (int i = 0; i < state_dim_; ++i)
        {
            // Simple update rule
            next_state[i] += (action == i) ? 0.1f : -0.05f;
            // Clip to [-2, 2]
            next_state[i] = std::max(-2.0f, std::min(2.0f, next_state[i]));
        }

        // Reward: negative sum of squares (want state near zero)
        float reward = 0.0f;
        for (int i = 0; i < state_dim_; ++i)
        {
            reward -= next_state[i] * next_state[i];
        }
        reward *= 0.1f; // Scale reward

        // Episode ends after 200 steps or if state diverges
        bool done = (step_count_ >= 200);
        for (int i = 0; i < state_dim_; ++i)
        {
            if (std::abs(next_state[i]) > 1.5f)
            {
                done = true;
                reward -= 10.0f; // Penalty for diverging
                break;
            }
        }

        state_ = next_state;

        return {next_state, reward, done};
    }

    const std::vector<float> &get_state() const { return state_; }

private:
    int state_dim_;
    int action_dim_;
    std::vector<float> state_;
    int step_count_;
};

int main()
{
    std::cout << "=== DQN Training Example ===" << std::endl
              << std::endl;

    // Environment parameters
    const int state_dim = 4;
    const int action_dim = 2;

    // Training parameters
    const int num_episodes = 100;
    const int warmup_steps = 1000; // Collect random experiences first
    const int train_freq = 4;      // Train every N steps

    // Create DQN agent
    DQN agent(state_dim, action_dim,
              0.001f, // learning_rate
              0.99f,  // gamma
              1.0f,   // epsilon_start
              0.01f,  // epsilon_end
              0.995f, // epsilon_decay
              10000,  // buffer_capacity
              32,     // batch_size
              100);   // target_update_freq

    // Create environment
    SimpleEnvironment env(state_dim, action_dim);

    // Statistics
    std::vector<float> episode_rewards;
    std::vector<int> episode_lengths;

    int total_steps = 0;

    std::cout << "Starting training..." << std::endl;
    std::cout << "Warmup steps: " << warmup_steps << std::endl
              << std::endl;

    // Training loop
    for (int episode = 0; episode < num_episodes; ++episode)
    {
        env.reset();
        const auto &state = env.get_state();

        float episode_reward = 0.0f;
        int episode_length = 0;
        bool done = false;

        while (!done)
        {
            // Select action
            bool training_mode = (total_steps >= warmup_steps);
            int action = agent.select_action(state.data(), training_mode);

            // Take action in environment
            auto [next_state, reward, is_done] = env.step(action);
            done = is_done;

            // Store experience
            agent.store_experience(state.data(), action, reward,
                                   next_state.data(), done);

            // Train agent
            if (total_steps >= warmup_steps && total_steps % train_freq == 0)
            {
                agent.train_step();
            }

            episode_reward += reward;
            episode_length++;
            total_steps++;
        }

        // Decay epsilon after each episode
        agent.decay_epsilon();

        // Record statistics
        episode_rewards.push_back(episode_reward);
        episode_lengths.push_back(episode_length);

        // Print progress
        if ((episode + 1) % 10 == 0)
        {
            float avg_reward = 0.0f;
            float avg_length = 0.0f;
            int window = std::min(10, (int)episode_rewards.size());

            for (int i = 0; i < window; ++i)
            {
                avg_reward += episode_rewards[episode_rewards.size() - 1 - i];
                avg_length += episode_lengths[episode_lengths.size() - 1 - i];
            }
            avg_reward /= window;
            avg_length /= window;

            std::cout << "Episode " << episode + 1 << "/" << num_episodes << std::endl;
            std::cout << "  Avg Reward (last 10): " << avg_reward << std::endl;
            std::cout << "  Avg Length (last 10): " << avg_length << std::endl;
            std::cout << "  Epsilon: " << agent.get_epsilon() << std::endl;
            std::cout << "  Total Steps: " << total_steps << std::endl;
            std::cout << "  Avg Loss: " << agent.get_avg_loss() << std::endl;
            std::cout << std::endl;
        }
    }

    std::cout << "=== Training Complete ===" << std::endl
              << std::endl;

    // Final statistics
    float total_reward = 0.0f;
    for (float r : episode_rewards)
    {
        total_reward += r;
    }

    std::cout << "Final Statistics:" << std::endl;
    std::cout << "  Total Episodes: " << num_episodes << std::endl;
    std::cout << "  Total Steps: " << total_steps << std::endl;
    std::cout << "  Average Reward: " << total_reward / num_episodes << std::endl;
    std::cout << "  Final Epsilon: " << agent.get_epsilon() << std::endl;

    // Test learned policy (greedy)
    std::cout << std::endl
              << "=== Testing Learned Policy ===" << std::endl;
    const int test_episodes = 10;
    float test_reward_total = 0.0f;

    for (int ep = 0; ep < test_episodes; ++ep)
    {
        env.reset();
        float test_reward = 0.0f;
        bool done = false;
        int steps = 0;

        while (!done && steps < 200)
        {
            const auto &state = env.get_state();
            int action = agent.select_action(state.data(), false); // Greedy
            auto [next_state, reward, is_done] = env.step(action);
            test_reward += reward;
            done = is_done;
            steps++;
        }

        test_reward_total += test_reward;
        std::cout << "Test Episode " << ep + 1 << ": Reward = "
                  << test_reward << ", Steps = " << steps << std::endl;
    }

    std::cout << std::endl
              << "Average Test Reward: "
              << test_reward_total / test_episodes << std::endl;

    return 0;
}
