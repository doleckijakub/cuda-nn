#pragma once

#include <iostream>
#include <vector>

#ifndef NNCU_ASSERT
#include <cassert>
#define NNCU_ASSERT assert
#endif // NNCU_ASSERT

#define NNCU_PRINT(obj) (obj).print(std::cout, #obj)

namespace nncu {

class NeuralNetwork;

class Matrix {
	size_t rows;
	size_t cols;
	float *elements;

public:

	Matrix(size_t rows, size_t cols);
	~Matrix();

	float &at(size_t i, size_t j);

	void fill(float x);
	void randomize(float low, float high);

	void operator+=(Matrix &);
	void dot(Matrix &a, Matrix &b);

	void activate();
	void shuffleRows();

	void print(std::ostream &sink, const char *name, const char *padding = "");

	friend class NeuralNetwork;

	friend void backpropagate(NeuralNetwork &network, NeuralNetwork &gradient, std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &outputs);
	friend void learn(NeuralNetwork &network, NeuralNetwork &gradient, float rate);
};

class NeuralNetwork {
	const std::vector<size_t> architecture;
	size_t size;

	std::vector<Matrix> weights;
	std::vector<Matrix> biasses;
	std::vector<Matrix> activations;

	size_t inputSize();
	size_t outputSize();

	std::vector<std::vector<float>> trainingInput;
	std::vector<std::vector<float>> trainingOutput;

public:

	class Layer {
		size_t size;
		float *data;

		Layer(size_t size, float *data);

	public:

		float &operator[](size_t index);

		friend class NeuralNetwork;
	};

	NeuralNetwork(std::vector<size_t> architecture);

	void remember(std::vector<float> input, std::vector<float> output);

	float cost();

	void train();

	void feed(std::vector<float> input);

	Layer output();

	void print(std::ostream &sink, const char *name);

	friend void backpropagate(NeuralNetwork &network, NeuralNetwork &gradient, std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &outputs);
	friend void learn(NeuralNetwork &network, NeuralNetwork &gradient, float rate);
};

}