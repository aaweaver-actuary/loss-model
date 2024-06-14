#ifndef CALCULATE_MU_HPP
#define CALCULATE_MU_HPP

#include <cmath>
#include <include/

// Declaration of the calculate_mu function for double types
double calculate_mu(double log_prem, double alpha, double beta, double ratio);

// Declaration of the calculate_mu function for Stan var types
stan::math::var calculate_mu(stan::math::var log_prem, stan::math::var alpha,
                             stan::math::var beta, stan::math::var ratio);

#endif // CALCULATE_MU_HPP
