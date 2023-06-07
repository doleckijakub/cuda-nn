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

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), stride(cols) {
	elements = new float[rows * cols];
}

Matrix::~Matrix() {
	delete[] elements;
}

float &Matrix::at(size_t i, size_t j) {
	return elements[i * stride + j];
}

float &Matrix::Row::at(size_t i, size_t j) {
	return elements[i * stride + j];
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

Matrix::Row::Row(Matrix &matrix, size_t row) {
	cols = matrix.cols;
	stride = matrix.stride;
	elements = &matrix.at(row, 0);
}

Matrix::Row Matrix::row(size_t row) {
	return Matrix::Row(*this, row);
}

Matrix Matrix::copy() {
	Matrix dst(rows, cols);
	for(size_t i = 0; i < rows; ++i) {
		for(size_t j = 0; j < cols; ++j) {
			dst.at(i, j) = at(i, j);
		}
	}
	return dst;
}

Matrix dot(Matrix &a, Matrix &b) {
	NNCU_ASSERT(a.cols == b.rows);
	size_t n = a.cols;

	Matrix dst(a.rows, b.cols);

	for(size_t i = 0; i < dst.rows; ++i) {
		for(size_t j = 0; j < dst.cols; ++j) {
			dst.at(i, j) = 0;
			for(size_t k = 0; k < n; ++k) {
				dst.at(i, j) += a.at(i, k) * b.at(k, j);
			}
		}
	}

	return dst;
}

Matrix Matrix::operator+(Matrix &other) {
	Matrix dst(rows, cols);
	for(size_t i = 0; i < rows; ++i) {
		for(size_t j = 0; j < cols; ++j) {
			dst.at(i, j) = other.at(i, j) + at(i, j);
		}
	}
	return dst;
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

void Matrix::print(std::ostream &sink) {
	sink << std::fixed << std::setprecision(5) << "[" << std::endl;
	for(size_t i = 0; i < rows; ++i) {
		sink << "\t";
		for(size_t j = 0; j < cols; ++j) {
			sink << at(i, j) << " ";
		}
		sink << std::endl;
	}
	sink << "]" << std::endl;
}

}