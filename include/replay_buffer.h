#pragma once
#include <vector>

struct Experience
{
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> next_state;
    bool done;
};

class ReplayBuffer
{
public:
    ReplayBuffer(size_t capacity);
    void add(const Experience &exp);
    std::vector<Experience> sample(size_t batch_size);
    size_t size() const;
    bool is_full() const;

private:
    std::vector<Experience> buffer_;
    size_t capacity_;
    size_t position_;
};