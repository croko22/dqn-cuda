#include "../include/dqn.h"
#include "../include/training_logger.h"
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

void run_training(bool use_double_dqn, const std::string &output_basename)
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

    // Use TrainingLogger
    TrainingLogger logger(output_basename);
    
    int num_episodes = 500;
    long global_step = 0;
    
    for (int episode = 0; episode < num_episodes; ++episode)
    {
        env.reset();
        float total_reward = 0.0f;
        int steps = 0;
        float episode_loss_sum = 0.0f;
        int train_count = 0;
        
        while (true)
        {
            int action = agent.select_action(env.get_state());
            
            float next_state[4];
            float reward;
            bool done;
            
            env.step(action, next_state, reward, done);
            
            agent.store_experience(env.get_state(), action, reward, next_state, done);
            agent.train_step();
            
            // Accumulate loss
            float loss = agent.get_avg_loss();
            if (loss > 0.0f) {
                episode_loss_sum += loss;
                train_count++;
            }
            
            total_reward += reward;
            steps++;
            global_step++;
            
            // Log step data
            logger.log_step(global_step, reward, loss, agent.get_last_q_value(), action);
            
            if (done)
                break;
        }
        
        agent.decay_epsilon();
        
        float avg_loss = (train_count > 0) ? episode_loss_sum / train_count : 0.0f;
        
        if (episode % 10 == 0)
        {
            std::cout << "Episode " << episode 
                      << " | Reward: " << total_reward 
                      << " | Loss: " << avg_loss 
                      << " | Epsilon: " << agent.get_epsilon() << std::endl;
            logger.flush();
        }
        
        logger.log_episode(episode, total_reward, steps, agent.get_epsilon(), avg_loss);
    }
    
    std::cout << "Training finished." << std::endl;
    
    // Save model using the same timestamped name
    std::string model_file = logger.get_base_filename() + ".bin";
    agent.save_model(model_file);
    std::cout << "Model saved to " << model_file << std::endl << std::endl;
}

int main()
{
    // Train Standard DQN
    run_training(false, "results/dqn_results");

    // Train Double DQN
    run_training(true, "results/double_dqn_results");

    return 0;
}
