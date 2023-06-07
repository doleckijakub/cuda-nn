CPP := g++
CPPFLAGS := -Wall -Wextra -pedantic -std=c++17 -I src
LDFLAGS := -lm

ifeq ($(OS),Windows_NT)
	WINDOWS := 1
else ifeq ($(shell uname),Linux)
	LINUX := 1
else ifeq ($(shell uname),Darwin)
	MACOS := 1
else
	$(error Unsupported platform: $(shell uname))
endif

all: example

example: example.cpp src/*
	$(CPP) $(CPPFLAGS) -o example example.cpp src/*.cpp $(LDFLAGS)

clean:
	git clean -fX