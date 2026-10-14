# JPEG-like image compression library - build file
#   make            build library, CLI, benchmark and tests into $(BUILD)/
#   make test       build and run the C++ unit tests
#   make clean
#
# Override on the command line if needed, e.g.
#   make CXX=g++-14 OPT="-O3 -march=native" BUILD=build_native
#   make OPT="-O2 -pg -fno-inline" LDFLAGS=-pg BUILD=build_prof      (gprof profile build)
BUILD    ?= build
CXX      ?= g++
OPT      ?= -O3
CXXFLAGS ?= -std=c++17 $(OPT) -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS  ?=

SRC   := $(wildcard src/*.cpp)
OBJ   := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRC))
LIB   := $(BUILD)/libjlc.a

all: $(LIB) $(BUILD)/jlc $(BUILD)/jlc_bench $(BUILD)/jlc_tests

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIB): $(OBJ)
	ar rcs $@ $^

$(BUILD)/jlc: apps/jlc_cli.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

$(BUILD)/jlc_bench: apps/jlc_bench.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

$(BUILD)/jlc_tests: tests/test_main.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

test: $(BUILD)/jlc_tests
	./$(BUILD)/jlc_tests

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
