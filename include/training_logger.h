#pragma once
#include <string>
#include <fstream>
#include <vector>

// Simple CSV logger for training metrics
class TrainingLogger
{
public:
    TrainingLogger(const std::string &filename);
    ~TrainingLogger();

    // Log episode metrics
    void log_episode(int episode, float reward, int steps, float epsilon, float avg_loss);

    // Log step metrics (more detailed)
    void log_step(int step, float reward, float loss, float q_value, int action);

    // Flush to disk
    void flush();

    // Get the base filename (including timestamp)
    std::string get_base_filename() const { return base_filename_; }

private:
    std::ofstream episode_file_;
    std::ofstream step_file_;
    std::string base_filename_;
};
