#include "nn.hpp"

#include <algorithm>
#include <iomanip>
#include <cmath>

namespace nncu {

float sigmoidf(float x) {
	return 1.f / (1.f + expf(-x));
}

float rand_float() {
	return (float) rand() / (float) RAND_MAX;
}

float rand_float_range(float low, float high) {
	return rand_float() * (high - low) + low;
}

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
	elements = new float[rows * cols];
}

Matrix::~Matrix() {
	delete[] elements;
}

float &Matrix::at(size_t i, size_t j) {
	NNCU_ASSERT(i < rows);
	NNCU_ASSERT(j < cols);
	return elements[i * cols + j];
}

void Matrix::fill(float x) {
	for(size_t i = 0; i < rows; ++i) {
		for(size_t j = 0; j < cols; ++j) {
			at(i, j) = x;
		}
	}
}

void Matrix::randomize(float low, float high) {
	for(size_t i = 0; i < rows; ++i) {
		for(size_t j = 0; j < cols; ++j) {
			at(i, j) = rand_float_range(low, high);
		}
	}
}

void Matrix::dot(Matrix &a, Matrix &b) {
	NNCU_ASSERT(rows == a.rows);
	NNCU_ASSERT(cols == b.cols);

	NNCU_ASSERT(a.cols == b.rows);
	size_t n = a.cols;

	for(size_t i = 0; i < rows; ++i) {
		for(size_t j = 0; j < cols; ++j) {
			at(i, j) = 0;
			for(size_t k = 0; k < n; ++k) {
				at(i, j) += a.at(i, k) * b.at(k, j);
			}
		}
	}
}

void Matrix::operator+=(Matrix &other) {
	NNCU_ASSERT(rows == other.rows);
	NNCU_ASSERT(cols == other.cols);

	for(size_t i = 0; i < rows; ++i) {
		for(size_t j = 0; j < cols; ++j) {
			at(i, j) += other.at(i, j);
		}
	}
}

void Matrix::activate() {
	for(size_t i = 0; i < rows; ++i) {
		for(size_t j = 0; j < cols; ++j) {
			at(i, j) = sigmoidf(at(i, j));
		}
	}
}

void Matrix::shuffleRows() {
	for(size_t i = 0; i < rows; ++i) {
		size_t j = i + rand() % (rows - i);
		if(i != j) {
			for(size_t k = 0; k < cols; ++k) {
				std::swap(at(i, k), at(j, k));
			}
		}
	}
}

void Matrix::print(std::ostream &sink, const char *name, const char *padding) {
	sink << std::fixed << std::setprecision(5);
	sink << padding << name << " = [" << std::endl;
	for(size_t i = 0; i < rows; ++i) {
		sink << padding << "    ";
		for(size_t j = 0; j < cols; ++j) {
			sink << at(i, j) << " ";
		}
		sink << std::endl;
	}
	sink << padding << "]" << std::endl;
}

NeuralNetwork::Layer::Layer(size_t size, float *data) : size(size), data(data) {}

float &NeuralNetwork::Layer::operator[](size_t index) {
	NNCU_ASSERT(index < size);
	return data[index];
}

size_t NeuralNetwork::inputSize() {
	return architecture[0];
}

size_t NeuralNetwork::outputSize() {
	return architecture[size];
}

NeuralNetwork::NeuralNetwork(std::vector<size_t> architecture) : architecture(architecture) {
	size = architecture.size() - 1;

	weights.reserve(size);
	biasses.reserve(size);
	activations.reserve(size + 1);

	activations.emplace_back(1, architecture[0]);

	for(size_t i = 1; i <= size; ++i) {
		weights.emplace_back(activations[i - 1].cols, architecture[i]).randomize(-1.f, 1.f);
		biasses.emplace_back(1, architecture[i]).randomize(-1.f, 1.f);
		activations.emplace_back(1, architecture[i]);
	}
}

void NeuralNetwork::remember(std::vector<float> input, std::vector<float> output) {
	NNCU_ASSERT(input.size() == inputSize());
	NNCU_ASSERT(output.size() == outputSize());
	trainingInput.push_back(input);
	trainingOutput.push_back(output);
}

float NeuralNetwork::cost() {
	size_t n = trainingInput.size();
	size_t m = outputSize();

	float c = 0;
	for (size_t i = 0; i < n; ++i) {
		feed(trainingInput[i]);
		for (size_t j = 0; j < m; ++j) {
			float d = output()[j] - trainingOutput[i][j];
			c += d * d;
		}
	}

	return c / n;
}

void backpropagate(NeuralNetwork &network, NeuralNetwork &gradient, std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &outputs) {
	size_t n = inputs.size();
	for(size_t i = 0; i < n; ++i) {
		for(size_t j = 0; j < n; ++i) {
			network.activations[0].at(0, j) = inputs[i][j];
		}

		network.feed(inputs[i]);

		for(size_t j = 0; j <= network.size; ++j) {
			gradient.activations[i].fill(0);
		}

		for(size_t j = 0; j < outputs[i].size(); ++j) {
			gradient.output()[j] = (network.output()[j] - outputs[i][j]) * 2 / n;
		}

		for(size_t l = network.size; l > 0; --l) {
			for(size_t j = 0; j < network.activations[l].cols; ++j) {
				float a = network.activations[l].at(0, j);
				float da = gradient.activations[l].at(0, j);

				gradient.biasses[l - 1].at(0, j) += da * a * (1 - a);

				for(size_t k = 0; k < network.activations[l - 1].cols; ++k) {
					float pa = network.activations[l - 1].at(0, k);
					float w = network.weights[l - 1].at(k, j);

					gradient.weights[l - 1].at(k, j) += da * a * (1 - a) * pa;
					gradient.activations[l - 1].at(0, k) += da * a * (1 - a) * w;
				}
			}
		}
	}
}

void learn(NeuralNetwork &network, NeuralNetwork &gradient, float rate) {
	for(size_t i = 0; i < network.size; ++i) {
		for(size_t j = 0; j < network.weights[i].rows; ++j) {
			for(size_t k = 0; j < network.weights[i].cols; ++k) {
				network.weights[i].at(j, k) -= rate * gradient.weights[i].at(j, k);
			}
		}

		for(size_t j = 0; j < network.biasses[i].rows; ++j) {
			for(size_t k = 0; j < network.biasses[i].cols; ++k) {
				network.biasses[i].at(j, k) -= rate * gradient.biasses[i].at(j, k);
			}
		}
	}
}

void NeuralNetwork::train() {
	static NeuralNetwork gradient(architecture);
	static float rate = 0.2f;
	
	for(auto &mat : gradient.weights) mat.fill(0);
	for(auto &mat : gradient.biasses) mat.fill(0);
	for(auto &mat : gradient.activations) mat.fill(0);

	backpropagate(*this, gradient, trainingInput, trainingOutput);
	learn(*this, gradient, rate);
}

void NeuralNetwork::feed(std::vector<float> input) {
	auto n = inputSize();
	NNCU_ASSERT(input.size() == n);
	for(size_t i = 0; i < n; ++i) {
		activations[0].at(0, i) = input[i];
	}
	for(size_t i = 0; i < size; ++i) {
		activations[i + 1].dot(activations[i], weights[i]);
		activations[i + 1] += biasses[i];
		activations[i + 1].activate();
	}
}

NeuralNetwork::Layer NeuralNetwork::output() {
	return NeuralNetwork::Layer(outputSize(), activations[activations.size() - 1].elements);
}

void NeuralNetwork::print(std::ostream &sink, const char *name) {
	sink << name << " = [" << std::endl;
	for(size_t i = 0; i < size; ++i) {
		weights[i].print(sink, (std::string("weights[") + std::to_string(i) + std::string("]")).c_str(), "    ");
		biasses[i].print(sink, (std::string("biasses[") + std::to_string(i) + std::string("]")).c_str(), "    ");
	}
	sink << "]" << std::endl;
}

}