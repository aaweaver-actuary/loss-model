#include "../src/calculate_mu.hpp"
#include <gtest/gtest.h>

// Test cases for the calculate_mu function
TEST(CalculateMuTest, PositiveInputs) {
  EXPECT_NEAR(calculate_mu(2.0, 1.0, 0.5, 0.3), 3.8, 1e-6);
}

TEST(CalculateMuTest, ZeroInputs) {
  EXPECT_NEAR(calculate_mu(0.0, 0.0, 0.0, 0.0), 0.0, 1e-6);
}

TEST(CalculateMuTest, NegativeInputs) {
  EXPECT_NEAR(calculate_mu(-2.0, -1.0, -0.5, -0.3), -3.8, 1e-6);
}

TEST(CalculateMuTest, MixedInputs) {
  EXPECT_NEAR(calculate_mu(2.0, -1.0, 0.5, -0.3), 1.2, 1e-6);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
