#pragma once

#include <iostream>

#ifndef NNCU_ASSERT
#include <cassert>
#define NNCU_ASSERT assert
#endif // NNCU_ASSERT

namespace nncu {

class Matrix {
	size_t rows;
	size_t cols;
	size_t stride;
	float *elements;

public:

	class Row {
		size_t cols;
		size_t stride;
		float *elements;

		Row(Matrix &matrix, size_t row);

	public:

		float &at(size_t i, size_t j);

		friend class Matrix;
	};

	Matrix(size_t rows, size_t cols);
	~Matrix();

	float &at(size_t i, size_t j);

	void fill(float x);
	void randomize(float low, float high);

	Row row(size_t row);

	Matrix copy();
	void operator+=(Matrix &);

	void activate();
	void shuffleRows();

	void print(std::ostream &sink);

	friend Matrix dot(Matrix &a, Matrix &b);
};

Matrix dot(Matrix &a, Matrix &b);

class NeuralNetwork {

};

}