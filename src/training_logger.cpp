#include "../include/training_logger.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>

TrainingLogger::TrainingLogger(const std::string &filename)
    : base_filename_(filename)
{
    // Create episode log file
    std::string episode_filename = filename + "_episodes.csv";
    episode_file_.open(episode_filename);
    if (!episode_file_.is_open())
    {
        std::cerr << "Failed to open episode log file: " << episode_filename << std::endl;
    }
    else
    {
        // Write header
        episode_file_ << "episode,reward,steps,epsilon,avg_loss\n";
        std::cout << "Logging episodes to: " << episode_filename << std::endl;
    }

    // Create step log file
    std::string step_filename = filename + "_steps.csv";
    step_file_.open(step_filename);
    if (!step_file_.is_open())
    {
        std::cerr << "Failed to open step log file: " << step_filename << std::endl;
    }
    else
    {
        // Write header
        step_file_ << "step,reward,loss,q_value,action\n";
        std::cout << "Logging steps to: " << step_filename << std::endl;
    }
}

TrainingLogger::~TrainingLogger()
{
    if (episode_file_.is_open())
    {
        episode_file_.close();
    }
    if (step_file_.is_open())
    {
        step_file_.close();
    }
}

void TrainingLogger::log_episode(int episode, float reward, int steps, float epsilon, float avg_loss)
{
    if (episode_file_.is_open())
    {
        episode_file_ << episode << ","
                      << std::fixed << std::setprecision(6) << reward << ","
                      << steps << ","
                      << std::fixed << std::setprecision(6) << epsilon << ","
                      << std::fixed << std::setprecision(6) << avg_loss << "\n";
    }
}

void TrainingLogger::log_step(int step, float reward, float loss, float q_value, int action)
{
    if (step_file_.is_open())
    {
        step_file_ << step << ","
                   << std::fixed << std::setprecision(6) << reward << ","
                   << std::fixed << std::setprecision(6) << loss << ","
                   << std::fixed << std::setprecision(6) << q_value << ","
                   << action << "\n";
    }
}

void TrainingLogger::flush()
{
    if (episode_file_.is_open())
    {
        episode_file_.flush();
    }
    if (step_file_.is_open())
    {
        step_file_.flush();
    }
}
