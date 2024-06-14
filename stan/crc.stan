data{
    int<lower=1> N;             // number of observations
    
    int<lower=1, upper=N> N_w;    // number of accident years
    int<lower=1, upper=N> N_d;    // number of development years
    
    array[N] int acc__raw; // accident year
    array[N] int dev; // development year

    vector[N] premium;              // premium for accident year
    vector[N] cum_rpt_loss;         // cumulative reported losses at cell [acc, dev]
}
transformed data{
    // accident year
    int min_acc = min(acc__raw);
    array[N] int<lower=1, upper=N_w> acc;
    for (i in 1:N){
        acc[i] = acc__raw[i] - min_acc + 1;
    } 

    // calendar year
    array[N] int cal;
    for (i in 1:N) {
        cal[i] = acc[i] + dev[i] - 1; 
    }
    int current_cal = max(acc);
    int next_cal = current_cal + 1;
    int N_current_cal = 0;
    int N_next_cal = 0;
    int N_future_cal = 0;
    for (i in 1:N) {
        if (cal[i] <= current_cal) N_current_cal += 1;
        else if (cal[i] == next_cal) {
            N_next_cal += 1;
            N_future_cal += 1;
        }
        else N_future_cal += 1;
    }

    // log of the premium
    vector[N] log_prem = log(premium);
    
    // incremental reported losses
    vector[N] inc_rpt_loss;                
    for (i in 1:N) {
        int d = dev[i];
        int w = acc[i];

        // if the development year is 1, the incremental reported loss is the same
        // as the cumulative reported loss
        if (d == 1) inc_rpt_loss[i] = cum_rpt_loss[i];
        else {
            // otherwise, look for the entry with the same accident year and the
            // previous development year
            for (j in 1:N) {
                if (acc[j] == w && dev[j] == d-1) {
                    // once you find them, the incremental reported loss is the difference
                    // between the cumulative reported losses at the two cells
                    inc_rpt_loss[i] = cum_rpt_loss[i] - cum_rpt_loss[j];
                }
            }
        }
    }

    // log of the cumulative reported losses
    vector[N] log_cum_rpt_loss;            
    for (i in 1:N) log_cum_rpt_loss[i] = log(cum_rpt_loss[i]);

    // acc and dev for current data only
    array[N_current_cal] int acc__current_cal;
    array[N_current_cal] int dev__current_cal;
    array[N_current_cal] real log_prem__current_cal;
    array[N_current_cal] real log_cum_rpt_loss__current_cal;
    array[N_current_cal] real inc_rpt_loss__current_cal;
    int j__current_cal = 1;
    for (i in 1:N) {
        if (cal[i] <= current_cal) {
            acc__current_cal[j__current_cal] = acc[i];
            dev__current_cal[j__current_cal] = dev[i];
            log_prem__current_cal[j__current_cal] = log_prem[i];
            log_cum_rpt_loss__current_cal[j__current_cal] = log_cum_rpt_loss[i];
            inc_rpt_loss__current_cal[j__current_cal] = inc_rpt_loss[i];
            j__current_cal += 1;
        }
    }
}
parameters{
    vector[N_w - 1] r_alpha__loss;      // random part of the alpha parameter for loss
    vector[N_d - 1] r_beta__rpt_loss;   // random part of the beta parameter for reported loss
    real log_elr;                       // log of the expected loss ratio
    vector<lower=0.0001>[N_d] sig2__rpt_loss;           // variance of the reported loss
}
transformed parameters{
    vector[N_w] alpha__loss;              // alpha parameter for loss
    vector[N_d] beta__rpt_loss;           // beta parameter for reported loss
    vector<lower=0>[N_d] sig__rpt_loss;   // standard deviation of the reported loss
    vector[N_current_cal] mu__rpt_loss;   // mean of the reported loss


    // set the first element of alpha to 0, and the remaining elements to
    // the random part of the alpha parameter (from parameters block)
    alpha__loss[1] = 0;
    for (i in 2:N_w) alpha__loss[i] = r_alpha__loss[i-1];

    // set the last element of beta to 0, and the remaining elements to
    for (i in 1:(N_d - 1)) beta__rpt_loss[i] = r_beta__rpt_loss[i];
    beta__rpt_loss[N_d] = 0;

    // the remaining elements are the sum of the shape parameters and the
    // shape parameter of the inverse gamma distribution
    for (i in 1:N_d) sig__rpt_loss[i] = sqrt(sig2__rpt_loss[i]);
    
    // Initialize the counter for mu__rpt_loss to ensure it only records the values for
    // the current triangle
    for (i in 1:N_current_cal){
        // Add mu at the `j__mu`-th position of the mu vector, using observed data from
        // the ith position of the observed data
        mu__rpt_loss[i] = log_prem__current_cal[i] + log_elr    // log of the expected ultimate loss (prem * elr)
                + alpha__loss[acc__current_cal[i]]              // alpha parameter for the accident year
                + beta__rpt_loss[dev__current_cal[i]];          // beta parameter for the development year
        
    }

}
model {
    r_alpha__loss ~ normal(0,3.162);
    r_beta__rpt_loss ~ normal(0,3.162);

    sig2__rpt_loss ~ inv_gamma(0.01,0.01);

    log_elr ~ normal(-.4,3.162);

    for (i in 1:N_current_cal) {
        log_cum_rpt_loss__current_cal[i] ~ normal(
            mu__rpt_loss[i],
            sig__rpt_loss[dev__current_cal[i]]
        );
    }
}
generated quantities{
    // Generate mu for all the data points
    vector[N] full_mu__rpt_loss;
    for (i in 1:N) {
        full_mu__rpt_loss[i] = log_prem[i] + log_elr // log of the expected ultimate loss (prem * elr)
                + alpha__loss[acc[i]]               // alpha parameter for the accident year
                + beta__rpt_loss[dev[i]];           // beta parameter for the development year
    }

    // Generate the log likelihood for all the data points
    vector[N] log_lik__rpt_loss;
    for (i in 1:N) log_lik__rpt_loss[i] = normal_lpdf(log_cum_rpt_loss[i]|full_mu__rpt_loss[i],sig__rpt_loss[dev[i]]);

    // Generate the cumulative reported losses predicted by the model
    vector[N] cum_rpt_loss__pred;
    for (i in 1:N) cum_rpt_loss__pred[i] = exp(normal_rng(full_mu__rpt_loss[i],sig__rpt_loss[dev[i]]));

    // Generate the residuals for the cumulative reported losses
    vector[N] cum_rpt_loss__residual;
    for (i in 1:N) cum_rpt_loss__residual[i] = cum_rpt_loss[i] - cum_rpt_loss__pred[i];

    // Generate the incremental reported losses predicted by the model
    vector[N] inc_rpt_loss__pred;
    for (i in 1:N) {
        int d = dev[i];
        int w = acc[i];

        if (d == 1) inc_rpt_loss__pred[i] = cum_rpt_loss__pred[i];
        else {
            for (j in 1:N) {
                if (acc[j] == w && dev[j] == d-1) {
                    inc_rpt_loss__pred[i] = cum_rpt_loss__pred[i] - cum_rpt_loss__pred[j];
                }
            }
        }
    }

    // Generate the residuals for the incremental reported losses
    vector[N] inc_rpt_loss__residual;
    for (i in 1:N) inc_rpt_loss__residual[i] = inc_rpt_loss[i] - inc_rpt_loss__pred[i];

    vector[N_w] generated_cum_rpt_loss__by_acc;
    for (i in 1:N_w) {
        // Cumulative amounts are the predicted cumulative amounts by acc for the current cal
        for (j in 1:N) {
            if ((acc[j] == i) && (cal[j] == current_cal)) {
                generated_cum_rpt_loss__by_acc[i] = exp(normal_rng(full_mu__rpt_loss[j],sig__rpt_loss[dev[j]]));
            }
        }
    }

    vector[N_w] generated_cum_rpt_loss__by_acc__next_cal;
    for (i in 1:N_w) {
        // Cumulative amounts are the predicted cumulative amounts by acc for the next cal
        for (j in 1:N) {
            if ((acc[j] == i) && (cal[j] == next_cal)) {
                generated_cum_rpt_loss__by_acc__next_cal[i] = exp(normal_rng(full_mu__rpt_loss[j],sig__rpt_loss[dev[j]]));
            }
        }
    }

    vector[N_w] generated_cum_rpt_loss__by_acc__ultimate;
    for (i in 1:N_w) {
        // Cumulative amounts are the predicted cumulative amounts by acc for the future cal
        for (j in 1:N) {
            if ((acc[j] == i) && (dev[j] == N_d)) {
                generated_cum_rpt_loss__by_acc__ultimate[i] = exp(normal_rng(full_mu__rpt_loss[j],sig__rpt_loss[dev[j]]));
            }
        }
    }

}
