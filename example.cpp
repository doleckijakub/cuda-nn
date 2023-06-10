#include <nn.hpp>

int main() {

	nncu::NeuralNetwork nn({ 2, 2, 1 });

	nn.remember({ 0, 0 }, { 0 });
	nn.remember({ 0, 1 }, { 1 });
	nn.remember({ 1, 0 }, { 1 });
	nn.remember({ 1, 1 }, { 0 });

	for(size_t i = 0; i < 20 * 1000; ++i) {
		nn.train();
		std::cout << i << ": cost = " << nn.cost() << std::endl;
	}

	for(float i : { 0.f, 1.f }) {
		for(float j : { 0.f, 1.f }) {
			nn.feed({ i, j });
			std::cout << i << " ^ " << j << " = " << nn.output()[0] << std::endl;
		}
	}

	return 0;
}