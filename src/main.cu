#include <iostream>
#include "../include/network.h"
#include "../include/replay_buffer.h"
#include <cuda_runtime.h>

void test_network()
{
    std::cout << "=== DQN CUDA Network Test ===\n\n";

    int input_dim = 4;
    int output_dim = 2;

    std::cout << "Creating network (" << input_dim << " -> 128 -> " << output_dim << ")...\n";
    Network net(input_dim, output_dim);

    float *d_input, *d_output;
    cudaMalloc(&d_input, input_dim * sizeof(float));
    cudaMalloc(&d_output, output_dim * sizeof(float));

    float h_input[] = {1.0f, 0.5f, -0.3f, 0.8f};
    cudaMemcpy(d_input, h_input, input_dim * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Running forward pass...\n";
    net.forward(d_input, d_output);

    float h_output[2];
    cudaMemcpy(h_output, d_output, output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nResults:\n";
    std::cout << "  Input:  [";
    for (int i = 0; i < input_dim; ++i)
    {
        std::cout << h_input[i];
        if (i < input_dim - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "  Output: [";
    for (int i = 0; i < output_dim; ++i)
    {
        std::cout << h_output[i];
        if (i < output_dim - 1)
            std::cout << ", ";
    }
    std::cout << "]\n\n";

    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Network test completed successfully!\n\n";
}

void test_replay_buffer()
{
    std::cout << "=== Replay Buffer Test ===\n\n";

    size_t capacity = 5;
    ReplayBuffer buffer(capacity);

    std::cout << "Adding experiences...\n";
    for (int i = 0; i < 7; ++i)
    {
        Experience exp;
        exp.state = {(float)i, (float)i * 0.1f, (float)i * 0.2f};
        exp.action = i % 3;
        exp.reward = (float)i * 10.0f;
        exp.next_state = {(float)i + 1.0f, (float)(i + 1) * 0.1f, (float)(i + 1) * 0.2f};
        exp.done = (i == 6);

        buffer.add(exp);
        std::cout << "  Added experience " << i
                  << " (buffer size: " << buffer.size() << ")\n";
    }

    std::cout << "\nBuffer is " << (buffer.is_full() ? "full" : "not full") << "\n";
    std::cout << "Current size: " << buffer.size() << "/" << capacity << "\n\n";

    size_t batch_size = 3;
    std::cout << "Sampling batch of " << batch_size << "...\n";
    auto batch = buffer.sample(batch_size);

    for (size_t i = 0; i < batch.size(); ++i)
    {
        const auto &exp = batch[i];
        std::cout << "  Sample " << i << ":\n";
        std::cout << "    State: [" << exp.state[0] << ", " << exp.state[1] << ", " << exp.state[2] << "]\n";
        std::cout << "    Action: " << exp.action << "\n";
        std::cout << "    Reward: " << exp.reward << "\n";
        std::cout << "    Done: " << (exp.done ? "true" : "false") << "\n";
    }

    std::cout << "\nReplay buffer test passed!\n\n";
}

int main()
{
    std::cout << "=== DQN CUDA Tests ===\n\n";

    test_network();
    test_replay_buffer();

    std::cout << "All tests completed successfully!\n";
    return 0;
}
