# Compiler
CXX = g++
CXXFLAGS = -std=c++11 -isystem /usr/include/gtest -pthread

# Directories
SRC_DIR = stan/src
TEST_DIR = stan/test
BUILD_DIR = stan/build
TEST_BUILD_DIR = $(BUILD_DIR)/test

# Google Test libraries
GTEST_LIBS = -lgtest -lgtest_main

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp)

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
TEST_OBJS = $(patsubst $(TEST_DIR)/%.cpp, $(TEST_BUILD_DIR)/%.o, $(TEST_SRCS))

# Main target executable
MAIN_TARGET = $(BUILD_DIR)/main

# Test target executable
TEST_TARGET = $(TEST_BUILD_DIR)/run_tests

# Ensure build directories exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_BUILD_DIR):
	mkdir -p $(TEST_BUILD_DIR)

# Default target
all: main test

# Link the main executable
$(MAIN_TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Link the test executable
$(TEST_TARGET): $(TEST_OBJS) $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(GTEST_LIBS)

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test files
$(TEST_BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp | $(TEST_BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(TEST_BUILD_DIR)

# Run tests
run_tests: $(TEST_TARGET)
	./$(TEST_TARGET)

.PHONY: all clean test main run_tests

# Targets
main: $(MAIN_TARGET)
test: $(TEST_TARGET)
