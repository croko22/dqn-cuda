#include "../include/dqn.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <random>

// Simple environment simulation (CartPole-like)
class SimpleEnv
{
public:
    SimpleEnv() : state_dim(4), action_dim(2) { reset(); }

    void reset()
    {
        // Random initial state
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.05f, 0.05f);
        
        for (int i = 0; i < state_dim; ++i)
            state[i] = dis(gen);
            
        steps = 0;
        done = false;
    }

    void step(int action, float *next_state, float &reward, bool &is_done)
    {
        // Simplified physics
        // 0: left, 1: right
        float force = (action == 1) ? 10.0f : -10.0f;
        
        // Update state (very simplified)
        state[0] += 0.01f * state[1]; // x += v
        state[1] += 0.01f * force;    // v += a
        state[2] += 0.01f * state[3]; // theta += omega
        state[3] += 0.01f * -force;   // omega += -force (simplified)

        // Check termination
        if (std::abs(state[0]) > 2.4f || std::abs(state[2]) > 0.209f || steps >= 200)
        {
            is_done = true;
        }
        else
        {
            is_done = false;
        }

        // Reward
        if (!is_done)
            reward = 1.0f;
        else if (steps >= 200)
            reward = 1.0f;
        else
            reward = 0.0f; // Fail

        // Copy next state
        for (int i = 0; i < state_dim; ++i)
            next_state[i] = state[i];
            
        done = is_done;
        steps++;
    }

    int get_state_dim() const { return state_dim; }
    int get_action_dim() const { return action_dim; }
    const float *get_state() const { return state; }

private:
    int state_dim;
    int action_dim;
    float state[4];
    int steps;
    bool done;
};

void run_training(bool use_double_dqn, const std::string &output_file)
{
    SimpleEnv env;
    int state_dim = env.get_state_dim();
    int action_dim = env.get_action_dim();

    std::cout << "Starting training with " << (use_double_dqn ? "Double DQN" : "DQN") << "..." << std::endl;

    DQN agent(state_dim, action_dim, 
              0.001f, // lr
              0.99f,  // gamma
              1.0f,   // epsilon_start
              0.01f,  // epsilon_end
              0.995f, // epsilon_decay
              10000,  // buffer_capacity
              64,     // batch_size
              10,     // target_update_freq
              use_double_dqn); // Enable/Disable DDQN

    std::ofstream log_file(output_file);
    log_file << "episode,steps,total_reward,avg_loss,epsilon\n";

    int num_episodes = 500;
    
    for (int episode = 0; episode < num_episodes; ++episode)
    {
        env.reset();
        float total_reward = 0.0f;
        
        while (true)
        {
            int action = agent.select_action(env.get_state());
            
            float next_state[4];
            float reward;
            bool done;
            
            env.step(action, next_state, reward, done);
            
            agent.store_experience(env.get_state(), action, reward, next_state, done);
            agent.train_step();
            
            total_reward += reward;
            
            if (done)
                break;
        }
        
        agent.decay_epsilon();
        
        if (episode % 10 == 0)
        {
            std::cout << "Episode " << episode 
                      << " | Reward: " << total_reward 
                      << " | Loss: " << agent.get_avg_loss() 
                      << " | Epsilon: " << agent.get_epsilon() << std::endl;
        }
        
        log_file << episode << "," 
                 << agent.get_steps() << "," 
                 << total_reward << "," 
                 << agent.get_avg_loss() << "," 
                 << agent.get_epsilon() << "\n";
    }
    
    log_file.close();
    std::cout << "Training finished. Results saved to " << output_file << std::endl;
    
    std::string model_file = output_file.substr(0, output_file.find_last_of('.')) + ".bin";
    agent.save_model(model_file);
    std::cout << "Model saved to " << model_file << std::endl << std::endl;
}

int main()
{
    // Train Standard DQN
    run_training(false, "dqn_results.csv");

    // Train Double DQN
    run_training(true, "double_dqn_results.csv");

    return 0;
}
