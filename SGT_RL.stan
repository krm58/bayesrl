// SGT Model:
// uses different learning rate distributions for different groups
// Because Stan will not allow for integer parameters, to specify two 
// different Beta priors on learning rates for the different groups, we will 
// put gamma priors on the parameters of the Beta prior
// KM adapted JMP code from this github repo: "https://github.com/jmxpearson/bayesrl"

data {
    int<lower = 0> N;  // number of observations
    int<lower = 0> Nsub;  // number of subjects
    int<lower = 1> Ncue;  // number of cues
    int<lower = 0> Ntrial;  // number of trials per subject
    int<lower = 0> Ngroup;  // number of experimental groups 
    int<lower = 0> sub[N];  // subject index 
    int<lower = 0> chosen[N];  // index of chosen option: 0 => missing
    int<lower = 1> trial[N];  // trial number
    int<lower = -4, upper = 4> outcomeFriend[N];  // outcome: -1 => missing
    int<lower = -4, upper = 4> outcomeSelf[N];  // outcome: -1 => missing
    int<lower = 0> group[Nsub];  // age group assignment for each subject, 0 => Adolescent, 2 => Adult
}

parameters {
    vector<lower = 0>[Nsub] beta;  // softmax parameter
    real<lower = 0, upper = 1> lambda[Nsub];  // learning rate, using notation from original SGT paper
    real<lower = 0> a[Ngroup];  // parameter for group-specific lambda, adolescents
    real<lower = 0> b[Ngroup];  // parameter for group-specific lambda, adults
    real<lower = 0> alpha[Nsub]; // "selfish" parameter, 1 = completely self-interested, 0 = completely friend-interested
}

transformed parameters {
    real V[Nsub, Ntrial, Ncue];  // value function for each deck
    real Deltaself[Nsub, Ntrial, Ncue];  // prediction error, self
    real Deltafriend[Nsub, Ntrial, Ncue]; //prediction error, friend
    real Qself[Nsub, Ntrial, Ncue]; // estimated rewards to self
    real Qfriend[Nsub, Ntrial, Ncue]; // estimated rewards to friend
    
    for (idx in 1:N) {
        if (trial[idx] == 1) {
            for (c in 1:Ncue) {
                V[sub[idx], trial[idx], c] <- 0;
                Qfriend[sub[idx], trial[idx], c] <- 0;
                Qself[sub[idx], trial[idx], c] <- 0;
                Deltafriend[sub[idx], trial[idx], c] <- 0;
                Deltaself[sub[idx], trial[idx], c] <- 0;
            }
        }
        if (trial[idx] < Ntrial) {  // push forward this trial's values
            for (c in 1:Ncue) {
                V[sub[idx], trial[idx] + 1, c] <- V[sub[idx], trial[idx], c];
                Qfriend[sub[idx], trial[idx] + 1, c] <- Qfriend[sub[idx], trial[idx], c];
                Qself[sub[idx], trial[idx] + 1, c] <- Qself[sub[idx], trial[idx], c];
                Deltafriend[sub[idx], trial[idx], c] <- 0;
                Deltaself[sub[idx], trial[idx], c] <- 0;
            }
            
            if (chosen[idx] > 0) {

                // prediction error, self
                Deltaself[sub[idx], trial[idx], chosen[idx]] <- outcomeSelf[idx] - Qself[sub[idx], trial[idx], chosen[idx]];

                // prediction error, friend
                Deltafriend[sub[idx], trial[idx], chosen[idx]] <- outcomeFriend[idx] - Qfriend[sub[idx], trial[idx], chosen[idx]];

                if (trial[idx] < Ntrial) {  // update action values for next trial
                    // update Qself
                    Qself[sub[idx], trial[idx] + 1, chosen[idx]] <- Qself[sub[idx], trial[idx], chosen[idx]] + lambda[sub[idx]] * Deltaself[sub[idx], trial[idx], chosen[idx]];

                    // update Qfriend
                    Qfriend[sub[idx], trial[idx] + 1, chosen[idx]] <- Qfriend[sub[idx], trial[idx], chosen[idx]] + lambda[sub[idx]] * Deltafriend[sub[idx], trial[idx], chosen[idx]];

                    // update V
                    V[sub[idx], trial[idx] + 1, chosen[idx]] <- alpha[sub[idx]] * Qself[sub[idx], trial[idx], chosen[idx]] + (1-alpha[sub[idx]])*Qfriend[sub[idx], trial[idx], chosen[idx]];
                }
                }
                
        }
        }
        }

model {
    beta ~ gamma(1, 0.2);
    alpha ~ uniform(0,1);
    a ~ gamma(1, 1);
    b ~ gamma(1, 1);


    for (idx in 1:Nsub) {
        lambda[idx] ~ beta(a[group[idx]], b[group[idx]]);
        chosen[idx] ~ categorical_logit(beta[sub[idx]] * to_vector(V[sub[idx], trial[idx]]));
    }

}

generated quantities {  // generate samples of learning rate from each group
    real<lower=0, upper=1> lambda_pred[Ngroup];
    vector[N] log_lik;
    for (grp in 1:Ngroup) {
        lambda_pred[grp] = beta_rng(a[grp], b[grp]);
    }

    for (idx in 1:N) {
        if (chosen[idx] > 0) {
            log_lik[idx] = categorical_logit_lpmf(chosen[idx] | beta[sub[idx]] * to_vector(V[sub[idx], trial[idx]]));
        }
    }
}