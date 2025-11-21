#include "../include/replay_buffer.h"
#include <algorithm>
#include <random>
#include <stdexcept>

ReplayBuffer::ReplayBuffer(size_t capacity)
    : capacity_(capacity), position_(0)
{
    buffer_.reserve(capacity);
}

void ReplayBuffer::add(const Experience &exp)
{
    if (buffer_.size() < capacity_)
    {
        buffer_.push_back(exp);
    }
    else
    {
        buffer_[position_] = exp;
    }
    position_ = (position_ + 1) % capacity_;
}

std::vector<Experience> ReplayBuffer::sample(size_t batch_size)
{
    if (batch_size > buffer_.size())
    {
        throw std::runtime_error("Batch size exceeds buffer size");
    }

    std::vector<Experience> batch;
    batch.reserve(batch_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, buffer_.size() - 1);

    for (size_t i = 0; i < batch_size; ++i)
    {
        size_t idx = dis(gen);
        batch.push_back(buffer_[idx]);
    }

    return batch;
}

size_t ReplayBuffer::size() const
{
    return buffer_.size();
}

bool ReplayBuffer::is_full() const
{
    return buffer_.size() == capacity_;
}
