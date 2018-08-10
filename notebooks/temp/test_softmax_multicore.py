import pymc3 as pm
print(pm.__version__)

import theano.tensor as tt
import theano
print(theano.__version__)

import patsy
import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    SEED = 20180727

    df = pd.read_csv(r'..\datasets\SoftmaxRegData1.csv', dtype={'Y':'category'})

    _, X = patsy.dmatrices('Y ~ 1 + standardize(X1) + standardize(X2)', data=df)

    # Number of categories
    n_cat = df.Y.cat.categories.size
    # Number of predictors
    n_pred = X.shape[1]

    with pm.Model() as model:
        
        ## `p`--quantity that I want to model--needs to have size (n_obs, n_cat). 
        ## Because `X` has size (n_obs, n_pred), then `beta` needs to have size (n_pred, n_cat)
        
        # priors for categories 1-2, excluding reference category 0 which is set to zero below (see DBDA2 p. 651 for explanation).   
        beta_ = pm.Normal('beta_', mu=0, sd=50, shape=(n_pred, n_cat-1))
        # add prior values zero for reference category 0. (add a column)  
        beta = pm.Deterministic('beta', tt.concatenate([tt.zeros((n_pred, 1)), beta_], axis=1))
        
        # The softmax function will squash the values in the range 0-1
        p = tt.nnet.softmax(tt.dot(np.asarray(X), beta))
        
        likelihood = pm.Categorical('likelihood', p=p, observed=df.Y.cat.codes.values)
        
        trace = pm.sample(draws=3000, tune=1000, chains=2, cores=4, random_seed=SEED)