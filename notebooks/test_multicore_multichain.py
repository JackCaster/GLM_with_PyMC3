import numpy as np
import pandas as pd

import theano
import pymc3 as pm

print('*** Start script ***')
print(f'{pm.__name__}: v. {pm.__version__}')
print(f'{theano.__name__}: v. {theano.__version__}')

if __name__ == '__main__':

    SEED = 20180730
    np.random.seed(SEED)

    # Generate data
    mu_real = 0
    sd_real = 1
    n_samples = 1000
    y = np.random.normal(loc=mu_real, scale=sd_real, size=n_samples)

    # Bayesian modelling
    with pm.Model() as model:
        
        mu = pm.Normal('mu', mu=0, sd=10)
        sd = pm.HalfNormal('sd', sd=10)
        
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sd=sd, observed=y)    
        trace = pm.sample(chains=2, cores=2, random_seed=SEED)

    print('Done!')

