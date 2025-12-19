#include "ev3_interface.hpp"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <stdexcept>

namespace ev3 {

// Helper para ejecutar comandos SSH
static std::string exec_ssh(const std::string& host, const std::string& ssh_key, const std::string& command) {
    // Expandir ~ si está presente
    std::string key_path = ssh_key;
    if (key_path[0] == '~') {
        const char* home = getenv("HOME");
        if (home) {
            key_path = std::string(home) + key_path.substr(1);
        }
    }
    
    std::string full_cmd = "ssh -i " + key_path + " -o StrictHostKeyChecking=no -o BatchMode=yes robot@" + host + " '" + command + "' 2>/dev/null";
    
    FILE* pipe = popen(full_cmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to execute SSH command");
    }
    
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    // Eliminar salto de línea final
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    
    return result;
}

EV3Interface::EV3Interface(const std::string& host, const std::string& ssh_key)
    : host(host), port(22), ssh_key_path(ssh_key) {
}

EV3Interface::~EV3Interface() {
    disconnect();
}

bool EV3Interface::connect() {
    try {
        // Verificar conexión ejecutando un comando simple
        std::string result = exec_ssh(host, ssh_key_path, "echo connected");
        return result == "connected";
    } catch (...) {
        return false;
    }
}

bool EV3Interface::isConnected() const {
    // Verificar con un ping rápido
    std::string cmd = "ping -c 1 -W 1 " + host + " > /dev/null 2>&1";
    return system(cmd.c_str()) == 0;
}

void EV3Interface::disconnect() {
    // No hay conexión persistente, cada comando es SSH individual
}

bool EV3Interface::setMotorSpeed(const std::string& motor_port, int speed) {
    try {
        // Agregar prefijo ev3-ports: si no lo tiene
        std::string full_port = motor_port;
        if (motor_port.find("ev3-ports:") == std::string::npos) {
            full_port = "ev3-ports:" + motor_port;
        }
        
        std::stringstream cmd;
        cmd << "for m in /sys/class/tacho-motor/motor*; do "
            << "if [ \\\"$(cat $m/address)\\\" = \\\"" << full_port << "\\\" ]; then "
            << "echo " << speed << " > $m/speed_sp; "
            << "echo run-forever > $m/command; "
            << "fi; done";
        
        exec_ssh(host, ssh_key_path, cmd.str());
        return true;
    } catch (...) {
        return false;
    }
}

bool EV3Interface::stopMotor(const std::string& motor_port) {
    try {
        // Agregar prefijo ev3-ports: si no lo tiene
        std::string full_port = motor_port;
        if (motor_port.find("ev3-ports:") == std::string::npos) {
            full_port = "ev3-ports:" + motor_port;
        }
        
        std::stringstream cmd;
        cmd << "for m in /sys/class/tacho-motor/motor*; do "
            << "if [ \\\"$(cat $m/address)\\\" = \\\"" << full_port << "\\\" ]; then "
            << "echo stop > $m/command; "
            << "fi; done";
        
        exec_ssh(host, ssh_key_path, cmd.str());
        return true;
    } catch (...) {
        return false;
    }
}

bool EV3Interface::stopAllMotors() {
    try {
        std::string cmd = "for m in /sys/class/tacho-motor/motor*/command; do echo stop > $m; done";
        exec_ssh(host, ssh_key_path, cmd);
        return true;
    } catch (...) {
        return false;
    }
}

GyroState EV3Interface::readGyro() {
    GyroState state;
    state.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    
    try {
        // Buscar el giroscopio y leer su valor
        std::string cmd = 
            "for s in /sys/class/lego-sensor/sensor*; do "
            "if grep -q gyro $s/driver_name 2>/dev/null; then "
            "cat $s/value0; exit 0; fi; done";
        
        std::string result = exec_ssh(host, ssh_key_path, cmd);
        state.angle = std::stoi(result);
        state.rate = 0; // TODO: leer rate si es necesario
        
    } catch (...) {
        state.angle = 0;
        state.rate = 0;
    }
    
    return state;
}

std::vector<int> EV3Interface::readSensor(const std::string& sensor_port) {
    std::vector<int> values;
    
    try {
        // Buscar el sensor en el puerto especificado y leer valores
        std::stringstream cmd;
        cmd << "for s in /sys/class/lego-sensor/sensor*; do "
            << "if [ \\\"$(cat $s/address)\\\" = \\\"ev3-ports:" << sensor_port << "\\\" ]; then "
            << "for i in 0 1 2 3 4 5 6 7; do "
            << "if [ -f $s/value$i ]; then cat $s/value$i; fi; "
            << "done; exit 0; fi; done";
        
        std::string result = exec_ssh(host, ssh_key_path, cmd.str());
        
        std::istringstream iss(result);
        std::string line;
        while (std::getline(iss, line)) {
            if (!line.empty()) {
                values.push_back(std::stoi(line));
            }
        }
        
    } catch (...) {
    }
    
    return values;
}

int EV3Interface::readUltrasonicSensor() {
    try {
        // Configurar el sensor en modo US-DIST-CM y leer distancia
        std::string cmd = 
            "for s in /sys/class/lego-sensor/sensor*; do "
            "if grep -q 'ev3-us' $s/driver_name 2>/dev/null; then "
            "echo US-DIST-CM > $s/mode 2>/dev/null; "
            "sleep 0.1; "
            "cat $s/value0; exit 0; fi; done";
        
        std::string result = exec_ssh(host, ssh_key_path, cmd);
        if (!result.empty()) {
            return std::stoi(result);
        }
        
    } catch (...) {
    }
    
    return 2550; // Distancia máxima por defecto
}

bool EV3Interface::setWheelSpeeds(int left_speed, int right_speed) {
    // Usar los motores grandes: outA (izquierda) y outD (derecha)
    bool left_ok = setMotorSpeed("outA", left_speed);
    bool right_ok = setMotorSpeed("outD", right_speed);
    return left_ok && right_ok;
}

GyroState EV3Interface::getBalanceState() {
    return readGyro();
}

bool EV3Interface::resetGyro() {
    try {
        std::string cmd = 
            "for s in /sys/class/lego-sensor/sensor*; do "
            "if grep -q gyro $s/driver_name 2>/dev/null; then "
            "echo GYRO-ANG > $s/mode; fi; done";
        
        exec_ssh(host, ssh_key_path, cmd);
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<std::string> EV3Interface::listMotors() {
    std::vector<std::string> motors;
    try {
        std::string cmd = "for m in /sys/class/tacho-motor/motor*/address; do cat $m; done";
        std::string result = exec_ssh(host, ssh_key_path, cmd);
        
        std::istringstream iss(result);
        std::string line;
        while (std::getline(iss, line)) {
            if (!line.empty()) {
                motors.push_back(line);
            }
        }
    } catch (...) {
    }
    return motors;
}

std::vector<std::string> EV3Interface::listSensors() {
    std::vector<std::string> sensors;
    try {
        std::string cmd = "for s in /sys/class/lego-sensor/sensor*/address; do cat $s; done";
        std::string result = exec_ssh(host, ssh_key_path, cmd);
        
        std::istringstream iss(result);
        std::string line;
        while (std::getline(iss, line)) {
            if (!line.empty()) {
                sensors.push_back(line);
            }
        }
    } catch (...) {
    }
    return sensors;
}

} // namespace ev3