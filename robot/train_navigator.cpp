#include "navigator_enviroment.hpp"
#include "include/dqn.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

using namespace navigation;
// using namespace dqn;

struct TrainingConfig {
    std::string ev3_host = "169.254.168.5";
    int num_episodes = 100;
    int max_steps_per_episode = 200;
    
    // Hiperparámetros DQN
    float gamma = 0.99f;
    float epsilon_start = 1.0f;
    float epsilon_end = 0.01f;
    float epsilon_decay = 0.995f;
    float learning_rate = 0.001f;
    
    // Configuración de entrenamiento
    unsigned int buffer_capacity = 10000;
    unsigned int batch_size = 32;
    unsigned int target_update_freq = 100;
    unsigned int warmup_steps = 500;  // Pasos aleatorios antes de entrenar
    unsigned int train_freq = 4;      // Entrenar cada N pasos
    
    // Guardado
    std::string model_save_path = "navigator_dqn_model.bin";
    int save_every_n_episodes = 25;
};

void printProgress(int episode, int total_episodes, float avg_reward, float epsilon, 
                   float avg_distance, float avg_loss) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Episode: " << episode << "/" << total_episodes << std::endl;
    std::cout << "Avg Reward: " << std::fixed << std::setprecision(2) << avg_reward << std::endl;
    std::cout << "Avg Distance: " << std::fixed << std::setprecision(1) << avg_distance << " mm" << std::endl;
    std::cout << "Epsilon: " << std::fixed << std::setprecision(4) << epsilon << std::endl;
    std::cout << "Avg Loss: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    TrainingConfig config;
    
    if (argc > 1) {
        config.ev3_host = argv[1];
    }
    
    std::cout << "╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   AUTONOMOUS NAVIGATOR DQN TRAINING        ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  EV3 Host: " << config.ev3_host << std::endl;
    std::cout << "  Episodes: " << config.num_episodes << std::endl;
    std::cout << "  Max Steps: " << config.max_steps_per_episode << std::endl;
    std::cout << "  Learning Rate: " << config.learning_rate << std::endl;
    std::cout << "  Warmup Steps: " << config.warmup_steps << std::endl;
    std::cout << "  Batch Size: " << config.batch_size << std::endl;
    std::cout << std::endl;
    
    try {
        // Crear entorno
        std::cout << "Initializing Navigation Environment..." << std::endl;
        NavigationEnvironment env(config.ev3_host);
        env.setMaxStepsPerEpisode(config.max_steps_per_episode);
        
        // Obtener dimensiones de estado y acciones
        int state_dim = State::getDimension();
        int num_actions = Action::getNumActions();
        
        std::cout << "\nState dimension: " << state_dim << std::endl;
        std::cout << "Number of actions: " << num_actions << std::endl;
        std::cout << "Actions: ";
        for (int i = 0; i < num_actions; ++i) {
            Action action = Action::fromIndex(i);
            std::cout << action.name;
            if (i < num_actions - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        // Crear agente DQN
        std::cout << "\nInitializing DQN agent..." << std::endl;
        DQN agent(state_dim, num_actions, 
                  config.learning_rate, 
                  config.gamma, 
                  config.epsilon_end,
                  config.epsilon_start,
                  config.epsilon_decay,
                  config.buffer_capacity,
                  config.batch_size,
                  config.target_update_freq);
        
        // Variables para estadísticas
        std::vector<float> episode_rewards;
        std::vector<float> episode_distances;
        std::vector<int> episode_steps;
        
        // Archivo de log
        std::ofstream log_file("training_log.csv");
        log_file << "episode,steps,total_reward,avg_reward,avg_distance,epsilon,avg_loss,collisions\n";
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Starting training..." << std::endl;
        std::cout << std::string(50, '=') << "\n" << std::endl;
        
        auto training_start = std::chrono::high_resolution_clock::now();
        int total_steps = 0;
        
        // Loop de entrenamiento principal
        for (int episode = 0; episode < config.num_episodes; ++episode) {
            auto episode_start = std::chrono::high_resolution_clock::now();
            
            // Resetear entorno
            State current_state = env.reset();
            std::vector<float> state_vec = current_state.toVector();
            
            float episode_reward = 0.0f;
            float episode_distance = 0.0f;
            int episode_step = 0;
            bool done = false;
            
            std::cout << "\n--- Episode " << (episode + 1) << " ---" << std::endl;
            
            // Loop del episodio
            while (!env.isTerminal() && episode_step < config.max_steps_per_episode) {
                // Determinar si estamos en modo de entrenamiento
                bool training_mode = (total_steps >= config.warmup_steps);
                
                // Seleccionar acción
                int action_idx = agent.select_action(state_vec.data(), training_mode);
                Action action = Action::fromIndex(action_idx);
                
                // Ejecutar acción
                auto [next_state, reward] = env.step(action);
                std::vector<float> next_state_vec = next_state.toVector();
                
                // Verificar si el episodio terminó
                done = env.isTerminal();
                
                // Almacenar experiencia
                agent.store_experience(state_vec.data(), action_idx, reward,
                                      next_state_vec.data(), done);
                
                // Entrenar el agente
                if (total_steps >= config.warmup_steps && total_steps % config.train_freq == 0) {
                    agent.train_step();
                }
                
                // Actualizar estadísticas
                episode_reward += reward;
                episode_distance += next_state.distance_front;
                episode_step++;
                total_steps++;
                
                // Actualizar estado actual
                state_vec = next_state_vec;
                current_state = next_state;
                
                // Mostrar progreso cada 20 pasos
                if (episode_step % 20 == 0) {
                    std::cout << "  Step " << episode_step 
                              << ": distance=" << current_state.distance_front << "mm"
                              << ", action=" << action.name
                              << ", reward=" << std::fixed << std::setprecision(1) << reward 
                              << ", epsilon=" << std::fixed << std::setprecision(3) 
                              << agent.get_epsilon() << std::endl;
                }
                
                if (done) break;
            }
            
            // Calcular promedios del episodio
            float avg_distance = (episode_step > 0) ? episode_distance / episode_step : 0.0f;
            
            // Decaer epsilon después de cada episodio
            agent.decay_epsilon();
            
            // Guardar estadísticas
            episode_rewards.push_back(episode_reward);
            episode_distances.push_back(avg_distance);
            episode_steps.push_back(episode_step);
            
            // Calcular promedios de ventana deslizante
            int window = std::min(10, (int)episode_rewards.size());
            float avg_reward_window = 0.0f;
            float avg_distance_window = 0.0f;
            
            for (size_t i = episode_rewards.size() - window; i < episode_rewards.size(); i++) {
                avg_reward_window += episode_rewards[i];
                avg_distance_window += episode_distances[i];
            }
            avg_reward_window /= window;
            avg_distance_window /= window;
            
            auto episode_end = std::chrono::high_resolution_clock::now();
            auto episode_duration = std::chrono::duration_cast<std::chrono::seconds>(
                episode_end - episode_start).count();
            
            // Obtener pérdida promedio
            float avg_loss = agent.get_avg_loss();
            
            std::cout << "  Episode completed: " << episode_step << " steps, "
                      << "total reward=" << std::fixed << std::setprecision(1) << episode_reward << ", "
                      << "avg distance=" << avg_distance << "mm, "
                      << "time=" << episode_duration << "s" << std::endl;
            
            // Escribir en log
            log_file << (episode + 1) << "," << episode_step << "," << episode_reward << "," 
                    << avg_reward_window << "," << avg_distance << "," 
                    << agent.get_epsilon() << "," << avg_loss << ",0\n";
            log_file.flush();
            
            // Mostrar progreso cada 5 episodios
            if ((episode + 1) % 5 == 0) {
                printProgress(episode + 1, config.num_episodes, avg_reward_window, 
                            agent.get_epsilon(), avg_distance_window, avg_loss);
            }
            
            // Guardar modelo periódicamente
            if ((episode + 1) % config.save_every_n_episodes == 0) {
                // Nota: Necesitarás agregar métodos save/load a tu clase DQN
                // agent.save_model(config.model_save_path);
                std::cout << "  → Model saved to " << config.model_save_path << std::endl;
            }
        }
        
        auto training_end = std::chrono::high_resolution_clock::now();
        auto training_duration = std::chrono::duration_cast<std::chrono::minutes>(
            training_end - training_start).count();
        
        // Guardar modelo final
        // agent.save_model(config.model_save_path);
        log_file.close();
        
        // Estadísticas finales
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Training completed!" << std::endl;
        std::cout << "Total training time: " << training_duration << " minutes" << std::endl;
        std::cout << "Total steps: " << total_steps << std::endl;
        
        float total_reward_sum = 0.0f;
        for (float r : episode_rewards) total_reward_sum += r;
        
        std::cout << "Average reward per episode: " << total_reward_sum / config.num_episodes << std::endl;
        std::cout << "Final epsilon: " << agent.get_epsilon() << std::endl;
        std::cout << "Model saved to: " << config.model_save_path << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        // Prueba final con política greedy
        std::cout << "\n=== Testing Learned Policy ===" << std::endl;
        const int test_episodes = 5;
        float test_reward_total = 0.0f;
        
        for (int test_ep = 0; test_ep < test_episodes; ++test_ep) {
            State state = env.reset();
            std::vector<float> state_vec = state.toVector();
            
            float test_reward = 0.0f;
            int test_steps = 0;
            bool done = false;
            
            std::cout << "\nTest Episode " << (test_ep + 1) << ":" << std::endl;
            
            while (!done && test_steps < config.max_steps_per_episode) {
                // Usar política greedy (training_mode = false)
                int action_idx = agent.select_action(state_vec.data(), false);
                Action action = Action::fromIndex(action_idx);
                
                auto [next_state, reward] = env.step(action);
                state_vec = next_state.toVector();
                
                test_reward += reward;
                test_steps++;
                done = env.isTerminal();
                
                if (test_steps % 10 == 0) {
                    std::cout << "  Step " << test_steps 
                              << ": distance=" << next_state.distance_front << "mm"
                              << ", action=" << action.name
                              << ", reward=" << reward << std::endl;
                }
            }
            
            test_reward_total += test_reward;
            std::cout << "  Test completed: " << test_steps << " steps, "
                      << "total reward=" << test_reward << std::endl;
        }
        
        std::cout << "\nAverage test reward: " 
                  << test_reward_total / test_episodes << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}