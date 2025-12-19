#ifndef NAVIGATION_ENVIRONMENT_HPP
#define NAVIGATION_ENVIRONMENT_HPP

#include "ev3_interface.hpp"
#include <vector>
#include <memory>

namespace navigation {

struct State {
    float distance_front;        // Distancia medida por el sensor ultrasónico
    float motor_speed_left;      // Velocidad del motor izquierdo
    float motor_speed_right;     // Velocidad del motor derecho
    float last_distance;         // Distancia anterior (para detectar acercamiento)
    int steps_without_collision; // Pasos desde la última colisión
    
    std::vector<float> toVector() const {
        return {distance_front, motor_speed_left, motor_speed_right, last_distance, 
                static_cast<float>(steps_without_collision)};
    }
    
    static int getDimension() { return 5; }
};

struct Action {
    enum Type { FORWARD, TURN_LEFT, TURN_RIGHT, BACKWARD, STOP };
    Type type;
    std::string name;
    int left_speed;
    int right_speed;
    
    static Action fromIndex(int idx) {
        switch(idx) {
            case 0: return {FORWARD, "FORWARD", 300, 300};
            case 1: return {TURN_LEFT, "TURN_LEFT", 100, 300};
            case 2: return {TURN_RIGHT, "TURN_RIGHT", 300, 100};
            case 3: return {BACKWARD, "BACKWARD", -200, -200};
            case 4: return {STOP, "STOP", 0, 0};
            default: return {STOP, "STOP", 0, 0};
        }
    }
    
    static int getNumActions() { return 5; }
};

class NavigationEnvironment {
private:
    std::unique_ptr<ev3::EV3Interface> ev3;
    
    State current_state;
    bool is_terminal;
    int step_count;
    
    // Parámetros del entorno
    static constexpr float COLLISION_THRESHOLD = 50.0f;  // mm
    int max_steps_per_episode;
    int total_steps_taken;
    
    // Para estadísticas
    float max_distance_reached;
    int collision_count;
    
    float calculateReward(const State& state, bool terminal);
    bool checkTerminal(const State& state);
    
public:
    NavigationEnvironment(const std::string& ev3_host);
    ~NavigationEnvironment();
    
    // Métodos del entorno RL
    void setMaxStepsPerEpisode(int steps) { max_steps_per_episode = steps; }
    State reset();
    std::pair<State, float> step(const Action& action);
    bool isTerminal() const { return is_terminal; }
    
    // Estado actual
    State getCurrentState() const { return current_state; }
    int getStepCount() const { return step_count; }
    
    // Estadísticas y verificación
    void printStats() const;
    void verifySensors();
    float readUltrasonicDistance();
};

} // namespace navigation

#endif // NAVIGATION_ENVIRONMENT_HPP