functions {
    /**
     * Calculate mu for a single observation.
      *
      * @param log_prem Log of the premium
      * @param alpha Alpha parameter
      * @param beta Beta parameter
      * @param ratio Ratio parameter
      * @return Mu
      */
    real calculate_mu(real log_prem, real alpha, real beta, real ratio) {
        return log_prem + ratio + alpha + beta;
    }

    /**
     * Transform accident years to start from 1, instead of the minimum accident year
     * in the data.
     *
     * @param N Number of observations
     * @param min_acc Minimum accident year
     * @param acc_raw Raw accident years
     * @return Transformed accident years
     */
    array[N] int transform_accident_years(int N, int min_acc, array[N] int acc_raw) {
        array[N] int acc;
        for (i in 1:N) {
            acc[i] = acc_raw[i] - min_acc + 1;
        }
        return acc;
    }

    /**
     * Calculate calendar years from accident and development years.
     *
     * @param N Number of observations
     * @param acc Accident years
     * @param dev Development years
     * @return Calendar years
     */
    array[N] int calculate_calendar_years(int N, array[N] int acc, array[N] int dev) {
        array[N] int cal;
        for (i in 1:N) {
            cal[i] = acc[i] + dev[i] - 1;
        }
        return cal;
    }

    /**
     * Calculate the log of the premium.
     *
     * @param N Number of observations
     * @param premium Premium
     * @return Log of the premium
     */
    vector[N] calculate_log_premium(int N, vector[N] premium) {
        return log(premium);
    }


    /**
     * Convert cumulative values to incremental values.
     *
     * @param N Number of observations
     * @param acc Accident years
     * @param dev Development years
     * @param cumulative Cumulative values to be converted
     * @return Incremental values
     */
    vector[N] cum_to_inc(int N, array[N] int acc, array[N] int dev, vector[N] cumulative) {
        vector[N] incremental;
        for (i in 1:N) {
            int d = dev[i];
            int w = acc[i];
            if (d == 1) {
                incremental[i] = cumulative[i];
            } else {
                for (j in 1:N) {
                    if (acc[j] == w && dev[j] == d-1) {
                        cumulative[i] = cumulative[i] - cumulative[j];
                    }
                }
            }
        }
        return incremental;
    }

    /**
     * Convert incremental values to cumulative values.
      *
      * @param N Number of observations
      * @param acc Accident years
      * @param dev Development years
      * @param incremental Incremental values to be converted
      * @return Cumulative values
      */
    vector[N] inc_to_cum(int N, array[N] int acc, array[N] int dev, vector[N] incremental) {
        vector[N] cumulative;
        for (i in 1:N) {
            int d = dev[i];
            int w = acc[i];
            if (d == 1) {
                cumulative[i] = incremental[i];
            } else {
                for (j in 1:N) {
                    if (acc[j] == w && dev[j] == d-1) {
                        cumulative[i] = cumulative[j] + incremental[i];
                    }
                }
            }
        }
        return cumulative;
    }

    
    /**
     * Calculate mu for the reporting loss.
     *
     * @param N Number of observations
     * @param N_w Number of distinct accident years
     * @param N_d Number of distinct development years
     * @param acc Vector of accident years (length N)
     * @param dev Vector of development years (length N)
     * @param log_prem Vector of the log of the premium (length N)
     * @param alpha Alpha parameters (length N_w - one for each accident year)
     * @param beta Beta parameters (length N_d - one for each development year)
     * @param ratio Ratio parameter (loss ratio, frequency, etc -- same for all observations)
     * @return full_mu for the reporting loss (length N)
     */        
    vector[N] calculate_full_mu(
      int N,
      int N_w,
      int N_d,
      array[N] int acc,
      array[N] int dev,
      vector[N] log_prem,
      vector[N_w] alpha,
      vector[N_d] beta,
      real ratio) {
        vector[N] full_mu;
        real a;
        real b;

        for (i in 1:N) {
            // Parameters needed for the calculate_mu function:
            a = alpha[acc[i]];
            b = beta[dev[i]];

            // Calculate mu for the observation:
            full_mu[i] = calculate_mu(log_prem[i], a, b, ratio);
        }
        return full_mu;
    }

    /**
     * Calculate loglikelihood for the model conditional on the data.
     *
     * @param N Number of observations
     * @param N_d Number of distinct development years
     * @param dev Array of development years (length N)
     * @param log_prem Vector of the log of the premium (length N)
     * @param alpha Alpha parameters (length N_w - one for each accident year)
     * @param beta Beta parameters (length N_d - one for each development year)
     * @param log_elr Log of the expected loss ratio
     * @return full_mu for the reporting loss (length N)
     */
    vector[N] log_likelihood(
      int N,
      int N_d,
      array[N] int dev,
      vector[N] log_cum_rpt_loss,
      vector[N] full_mu_rpt_loss,
      vector[N_d] sig_rpt_loss
      ) {
        vector[N] log_lik_rpt_loss;
        for (i in 1:N) {
            log_lik_rpt_loss[i] = normal_lpdf(log_cum_rpt_loss[i] | full_mu_rpt_loss[i], sig_rpt_loss[dev[i]]);
        }
        return log_lik_rpt_loss;
    }

    
    vector[N] generate_cum_rpt_loss_pred(int N, vector[N] full_mu_rpt_loss, vector[N] sig_rpt_loss) {
        vector[N] cum_rpt_loss_pred;
        for (i in 1:N) {
            cum_rpt_loss_pred[i] = exp(normal_rng(full_mu_rpt_loss[i], sig_rpt_loss[i]));
        }
        return cum_rpt_loss_pred;
    }

    vector[N] calculate_residuals(int N, vector[N] actual_loss, vector[N] predicted_loss) {
        vector[N] residuals;
        for (i in 1:N) {
            residuals[i] = actual_loss[i] - predicted_loss[i];
        }
        return residuals;
    }

    vector[N] generate_inc_rpt_loss_pred(int N, array[N] int acc, array[N] int dev, vector[N] cum_rpt_loss_pred) {
        vector[N] inc_rpt_loss_pred;
        for (i in 1:N) {
            int d = dev[i];
            int w = acc[i];
            if (d == 1) {
                inc_rpt_loss_pred[i] = cum_rpt_loss_pred[i];
            } else {
                for (j in 1:N) {
                    if (acc[j] == w && dev[j] == d-1) {
                        inc_rpt_loss_pred[i] = cum_rpt_loss_pred[i] - cum_rpt_loss_pred[j];
                    }
                }
            }
        }
        return inc_rpt_loss_pred;
    }
}

data {
    int<lower=1> N;
    int<lower=1, upper=N> N_w;
    int<lower=1, upper=N> N_d;
    array[N] int acc_raw;
    array[N] int dev;
    vector[N] premium;
    vector[N] cum_rpt_loss;
}

transformed data {
    int min_acc = min(acc_raw);
    array[N] int acc = transform_accident_years(N, min_acc, acc_raw);
    array[N] int cal = calculate_calendar_years(N, acc, dev);
    vector[N] log_prem = calculate_log_premium(N, premium);
    vector[N] inc_rpt_loss = cum_to_inc(N, acc, dev, cum_rpt_loss);
}

parameters {
    real log_elr;
    vector[N_w] alpha_loss;
    vector[N_d] beta_rpt_loss;
    vector<lower=0>[N_d] sig_rpt_loss;
}

model {
    vector[N] full_mu_rpt_loss = calculate_full_mu_rpt_loss(N, acc, dev, log_prem, alpha_loss, beta_rpt_loss, log_elr);
    target += sum(log_likelihood(N, log(cum_rpt_loss), full_mu_rpt_loss, sig_rpt_loss));
}

generated quantities {
    vector[N] full_mu_rpt_loss = calculate_full_mu_rpt_loss(N, acc, dev, log_prem, alpha_loss, beta_rpt_loss, log_elr);
    vector[N] cum_rpt_loss_pred = generate_cum_rpt_loss_pred(N, full_mu_rpt_loss, sig_rpt_loss);
    vector[N] cum_rpt_loss_residual = calculate_residuals(N, cum_rpt_loss, cum_rpt_loss_pred);
    vector[N] inc_rpt_loss_pred = generate_inc_rpt_loss_pred(N, acc, dev, cum_rpt_loss_pred);
    vector[N] inc_rpt_loss_residual = calculate_residuals(N, inc_rpt_loss, inc_rpt_loss_pred);
}
