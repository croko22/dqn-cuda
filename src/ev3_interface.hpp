#pragma once

#include <string>
#include <vector>
#include <memory>

namespace ev3 {

struct MotorState {
    std::string address;
    int position;
    int speed;
    std::string state;
};

struct SensorState {
    std::string address;
    std::string type;
    std::vector<int> values;
};

struct GyroState {
    int angle;          // Ángulo acumulado (grados)
    int rate;           // Velocidad angular (grados/s)
    long timestamp_ms;  // Timestamp
};

class EV3Interface {
private:
    std::string host;
    int port;
    std::string ssh_key_path;
    
public:
    EV3Interface(const std::string& host, const std::string& ssh_key = "~/.ssh/id_rsa_ev3");
    ~EV3Interface();
    
    // Métodos de conexión
    bool connect();
    bool isConnected() const;
    void disconnect();
    
    // Control de motores
    bool setMotorSpeed(const std::string& motor_port, int speed);
    bool stopMotor(const std::string& motor_port);
    bool stopAllMotors();
    MotorState getMotorState(const std::string& motor_port);
    
    // Lectura de sensores
    GyroState readGyro();
    std::vector<int> readSensor(const std::string& sensor_port);
    int readUltrasonicSensor();  // Específico para sensor ultrasónico
    
    // Para Gyro Boy específicamente
    bool setWheelSpeeds(int left_speed, int right_speed);
    GyroState getBalanceState();
    
    // Utilidades
    bool resetGyro();
    std::vector<std::string> listMotors();
    std::vector<std::string> listSensors();
};

} // namespace ev3
