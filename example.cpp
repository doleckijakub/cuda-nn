#include <nn.hpp>

int main(int argc, char const *argv[]) {
	(void) argc;
	(void) argv;

	nncu::Matrix mat(2, 5);
	mat.print(std::cout);

	return 0;
}