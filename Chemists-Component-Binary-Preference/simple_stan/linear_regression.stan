data {
  int<lower=0> N;        // Number of data points
  vector[N] x;           // Independent variable
  vector[N] y;           // Dependent variable
}

parameters {
  real alpha;            // Intercept
  real beta;             // Slope
  real<lower=0> sigma;   // Error scale
}

model {
  y ~ normal(alpha + beta * x, sigma);  // Likelihood
}
