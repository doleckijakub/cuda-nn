# CUDA nn

[![CI](https://github.com/doleckijakub/cuda-nn/actions/workflows/build.yml/badge.svg)](https://github.com/doleckijakub/cuda-nn/actions/workflows/build.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![c++17](https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c++)](https://en.cppreference.com/w/cpp/17)

## Usage

To add `cuda-nn` to your project just include `nn.hpp`.

```cpp
#include <nn.hpp>
```

To implement the neural network, create an object of class `nncu::NeuralNetwork`, specifying the architecture of the neural network as an initializer list.

```cpp
nncu::NeuralNetwork nn({ 2, 2, 1 });
```

To input the training data, use `nncu::NeuralNetwork::remeber`, passing the input vector and the desired output.

```cpp
nn.remember({ 0, 0 }, { 0 });
nn.remember({ 0, 1 }, { 1 });
nn.remember({ 1, 0 }, { 1 });
nn.remember({ 1, 1 }, { 0 });
```

To train the network, just use `nncu::NeuralNetwork::train`.

```cpp
nn.train();
```

You can get the [current cost](https://en.wikipedia.org/wiki/Loss_function) of the network, use  `nncu::NeuralNetwork::cost`.

```cpp
std::cout << nn.cost() << std::endl;
```

Once proficiently trained you can feed forward the input to the network using `nncu::NeuralNetwork::feed` and get the output using `nncu::NeuralNetwork::output`.

```cpp
nn.feed({ 0, 0 }); std::cout << nn.output()[0] << std::endl;
nn.feed({ 0, 1 }); std::cout << nn.output()[0] << std::endl;
nn.feed({ 1, 0 }); std::cout << nn.output()[0] << std::endl;
nn.feed({ 1, 1 }); std::cout << nn.output()[0] << std::endl;
```
