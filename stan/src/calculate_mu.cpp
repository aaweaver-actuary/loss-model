#include "calculate_mu.hpp"

// Definition of the calculate_mu function for double types
double calculate_mu(double log_prem, double alpha, double beta, double ratio) {
  return log_prem + ratio + alpha + beta;
}

// Definition of the calculate_mu function for Stan var types
stan::math::var calculate_mu(stan::math::var log_prem, stan::math::var alpha,
                             stan::math::var beta, stan::math::var ratio) {
  return log_prem + ratio + alpha + beta;
}
