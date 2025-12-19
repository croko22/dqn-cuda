#include "navigator_enviroment.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>

namespace navigation {

using namespace ev3;

NavigationEnvironment::NavigationEnvironment(const std::string& ev3_host)
    : is_terminal(false), step_count(0), max_steps_per_episode(200),
      total_steps_taken(0), max_distance_reached(0.0f), collision_count(0) {
    
    std::cout << "Connecting to EV3 at " << ev3_host << "..." << std::endl;
    
    ev3 = std::make_unique<EV3Interface>(ev3_host);
    
    if (!ev3->connect()) {
        throw std::runtime_error("Failed to connect to EV3 at " + ev3_host);
    }
    
    std::cout << "✓ Connected to EV3 at " << ev3_host << std::endl;
    
    // Verificar sensores
    verifySensors();
}

NavigationEnvironment::~NavigationEnvironment() {
    if (ev3) {
        ev3->stopAllMotors();
        std::cout << "\nMotors stopped." << std::endl;
    }
}

void NavigationEnvironment::verifySensors() {
    std::cout << "\n=== VERIFICANDO SENSORES ===" << std::endl;
    
    // Listar todos los sensores
    auto sensors = ev3->listSensors();
    std::cout << "Sensores detectados: " << sensors.size() << std::endl;
    for (const auto& sensor : sensors) {
        std::cout << "  - " << sensor << std::endl;
    }
    
    // Listar motores
    auto motors = ev3->listMotors();
    std::cout << "\nMotores detectados: " << motors.size() << std::endl;
    for (const auto& motor : motors) {
        std::cout << "  - " << motor << std::endl;
    }
    
    // Leer sensor ultrasónico
    float distance = readUltrasonicDistance();
    std::cout << "\nLectura sensor ultrasónico: " << distance << " mm" << std::endl;
    
    if (distance < 0 || distance > 3000) {
        std::cerr << "✗ ERROR: Lectura de sensor inválida!" << std::endl;
    } else {
        std::cout << "✓ Todos los sensores funcionando correctamente" << std::endl;
    }
    
    std::cout << "================================\n" << std::endl;
}

float NavigationEnvironment::readUltrasonicDistance() {
    return static_cast<float>(ev3->readUltrasonicSensor());
}

State NavigationEnvironment::reset() {
    // Detener motores
    ev3->stopAllMotors();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Leer distancia inicial
    float initial_distance = readUltrasonicDistance();
    
    // Resetear estado
    current_state.distance_front = initial_distance;
    current_state.motor_speed_left = 0;
    current_state.motor_speed_right = 0;
    current_state.last_distance = initial_distance;
    current_state.steps_without_collision = 0;
    
    is_terminal = false;
    step_count = 0;
    
    return current_state;
}

std::pair<State, float> NavigationEnvironment::step(const Action& action) {
    step_count++;
    total_steps_taken++;
    
    // Ejecutar acción
    ev3->setWheelSpeeds(action.left_speed, action.right_speed);
    
    // Esperar que se ejecute la acción
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Leer nuevo estado
    current_state.last_distance = current_state.distance_front;
    current_state.distance_front = readUltrasonicDistance();
    current_state.motor_speed_left = action.left_speed;
    current_state.motor_speed_right = action.right_speed;
    
    // Verificar colisión
    if (current_state.distance_front < COLLISION_THRESHOLD) {
        current_state.steps_without_collision = 0;
        collision_count++;
    } else {
        current_state.steps_without_collision++;
    }
    
    // Actualizar máxima distancia alcanzada
    if (current_state.distance_front > max_distance_reached) {
        max_distance_reached = current_state.distance_front;
    }
    
    // Verificar condición de término
    is_terminal = checkTerminal(current_state);
    
    // Calcular recompensa
    float reward = calculateReward(current_state, is_terminal);
    
    return {current_state, reward};
}

float NavigationEnvironment::calculateReward(const State& state, bool terminal) {
    float reward = 0.0f;
    
    // Penalización fuerte por colisión
    if (state.distance_front < COLLISION_THRESHOLD) {
        reward -= 100.0f;
        return reward;  // Retornar inmediatamente
    }
    
    // Recompensa por moverse
    if (state.motor_speed_left != 0 || state.motor_speed_right != 0) {
        reward += 1.0f;
    }
    
    // Recompensa por mantener distancia segura (100-500mm)
    if (state.distance_front > 100.0f && state.distance_front < 500.0f) {
        reward += 2.0f;
    }
    
    // Recompensa por evitar obstáculos (distancia aumentó)
    if (state.distance_front > state.last_distance) {
        reward += 3.0f;
    }
    
    // Penalización por acercarse demasiado
    if (state.distance_front < 100.0f) {
        reward -= 5.0f;
    }
    
    // Bono por muchos pasos sin colisión
    if (state.steps_without_collision > 20) {
        reward += 5.0f;
    }
    
    return reward;
}

bool NavigationEnvironment::checkTerminal(const State& state) {
    // Colisión
    if (state.distance_front < COLLISION_THRESHOLD) {
        std::cout << "  [TERMINAL] Colisión detectada!" << std::endl;
        return true;
    }
    
    // Máximo número de pasos
    if (step_count >= max_steps_per_episode) {
        std::cout << "  [TERMINAL] Máximo de pasos alcanzado" << std::endl;
        return true;
    }
    
    return false;
}

void NavigationEnvironment::printStats() const {
    std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        ESTADÍSTICAS DE NAVEGACIÓN          ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;
    std::cout << "Total steps: " << total_steps_taken << std::endl;
    std::cout << "Collisions: " << collision_count << std::endl;
    std::cout << "Max distance reached: " << max_distance_reached << " mm" << std::endl;
    std::cout << "=========================================\n" << std::endl;
}
} // namespace navigation