#include <vector>
#include <thread>
#include <mutex>
#include <iostream>
#include <fstream>
#include <sstream>

class Perceptron {
public:
    Perceptron(int input_size, double learning_rate)
        : weights(input_size + 1, 0.0), learning_rate(learning_rate) {}

    double predict(const std::vector<double>& inputs) {
        double activation = weights.back(); // bias term
        for (size_t i = 0; i < inputs.size(); ++i) {
            activation += weights[i] * inputs[i];
        }
        return activation >= 0.0 ? 1.0 : 0.0;
    }

    void train(const std::vector<std::vector<double>>& training_inputs, 
               const std::vector<double>& labels, 
               int epochs);

private:
    std::vector<double> weights;
    double learning_rate;
    std::mutex mtx;

    void update_weights(const std::vector<double>& inputs, double error);
};

void Perceptron::update_weights(const std::vector<double>& inputs, double error) {
    std::lock_guard<std::mutex> lock(mtx);
    for (size_t i = 0; i < inputs.size(); ++i) {
        weights[i] += learning_rate * error * inputs[i];
    }
    weights.back() += learning_rate * error; // Update bias term
}

void Perceptron::train(const std::vector<std::vector<double>>& training_inputs, 
                       const std::vector<double>& labels, 
                       int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<std::thread> threads;
        for (size_t i = 0; i < training_inputs.size(); ++i) {
            threads.emplace_back([this, &training_inputs, &labels, i]() {
                double prediction = predict(training_inputs[i]);
                double error = labels[i] - prediction;
                update_weights(training_inputs[i], error);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
}

std::vector<std::vector<double>> read_csv_inputs(const std::string& filename, int input_size) {
    std::vector<std::vector<double>> dataset;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::vector<double> inputs(input_size);
        std::stringstream ss(line);
        std::string value;
        for (int i = 0; i < input_size; ++i) {
            std::getline(ss, value, ',');
            inputs[i] = std::stod(value);
        }
        dataset.push_back(inputs);
    }

    return dataset;
}

std::vector<double> read_csv_labels(const std::string& filename) {
    std::vector<double> labels;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        labels.push_back(std::stod(line));
    }

    return labels;
}


int main() {
    std::string input_filename = "dataset_inputs.csv";
    std::string label_filename = "dataset_labels.csv";
    int input_size = 2; // Cambia esto según el número de características de tu dataset
    int epochs = 100;
    double learning_rate = 0.1;

    auto training_inputs = read_csv_inputs(input_filename, input_size);
    auto labels = read_csv_labels(label_filename);

    Perceptron perceptron(input_size, learning_rate);
    perceptron.train(training_inputs, labels, epochs);

    std::vector<double> *test_input = new std::vector<double>[10];
    test_input[0] = {0.5, 0.7};
    test_input[1] = {0.1, 0.2};
    test_input[2] = {0.3, 0.8};
    test_input[3] = {0.6, 0.9};
    test_input[4] = {0.2, 0.5};
    test_input[5] = {0.8, 0.1};
    test_input[6] = {0.4, 0.3};
    test_input[7] = {0.7, 0.6};
    test_input[8] = {0.9, 0.4};
    test_input[9] = {0.5, 0.5};

    for(int i = 0; i < 10; i++){
      double prediction = perceptron.predict(test_input[i]);
      std::cout << "Predicción para [" << test_input[i][0] << " ," << test_input[i][1] << " ] :" << prediction << std::endl; 
    }

    return 0;
}