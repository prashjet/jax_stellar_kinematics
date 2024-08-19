import jax.numpy as np
from jax.random import PRNGKey
import jax.scipy as jsp

import numpyro
numpyro.set_host_device_count(3)
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import dill
import pickle

from sanders_evans import LOSVD
losvd = LOSVD(pkernel_type='uniform')

def legendre(x): # cannot pass the poly order as argument, jax complains
    plgndr = jsp.special.lpmn_values(
        mdegree, 
        mdegree, 
        x, 
        is_normalized=False)
    return plgndr[0]

def get_model_spec(ssp_w, v, sigma, sigmaK, delta, kappa, mpolyweights, FZqmu, u, nfft, xpoly, losvd):
    Flosvd = losvd.evaluate_fft(u, v, sigma, sigmaK, delta, kappa)
    Fy = Flosvd * (np.sum(ssp_w) * FZqmu[:,0] + np.dot(FZqmu[:,1:], ssp_w))
    y = np.fft.irfft(Fy, nfft)
    leg_pols = legendre(xpoly)
    mpoly = np.sum(leg_pols.T * np.append(1, mpolyweights), 1)
    y = mpoly * y
    return y

def numpyro_model(lspec, lerr, FZqmu, u, nfft, xpoly, mask, q_pca, mdegree, losvd):
    ssp_w = numpyro.sample('ssp_w', dist.Normal(0.0,1.0), sample_shape=(q_pca,))
    v = numpyro.sample('v', dist.Uniform(low=400., high=800.))
    sigma = numpyro.sample('sigma', dist.HalfNormal(scale=100.))
    sigmaK = numpyro.sample('sigmaK', dist.HalfNormal(1.0))
    delta = numpyro.sample('delta', dist.Uniform(low=-losvd.delta_max, high=losvd.delta_max))
    kappa = numpyro.sample('kappa', dist.Uniform(low=-1., high=1.))
    mpolyweights = numpyro.sample('mpolyweights', dist.Normal(0.0,1.0), sample_shape=(mdegree,))
    ybar = get_model_spec(ssp_w, v, sigma, sigmaK, delta, kappa, mpolyweights, FZqmu, u, nfft, xpoly, losvd)
    # Masking with numpyro.handlers.mask doesn't work - chains don't move
    # with numpyro.handlers.mask(mask=mask):
    #     y = numpyro.sample('y', dist.Normal(ybar, lerr), obs=lspec)
    # ... use the mask manually instead
    y = numpyro.sample('y', dist.Normal(ybar[mask], lerr[mask]), obs=lspec[mask])
    return

def numpyro_model_for_predictive_distribution(lspec, lerr, FZqmu, u, nfft, xpoly, mask, q_pca, mdegree, losvd):
    ssp_w = numpyro.sample('ssp_w', dist.Normal(0.0,1.0), sample_shape=(q_pca,))
    v = numpyro.sample('v', dist.Uniform(low=400., high=800.))
    sigma = numpyro.sample('sigma', dist.HalfNormal(scale=100.))
    sigmaK = numpyro.sample('sigmaK', dist.HalfNormal(1.0))
    delta = numpyro.sample('delta', dist.Uniform(low=-losvd.delta_max, high=losvd.delta_max))
    kappa = numpyro.sample('kappa', dist.Uniform(low=-1., high=1.))
    mpolyweights = numpyro.sample('mpolyweights', dist.Normal(0.0,1.0), sample_shape=(mdegree,))
    ybar = get_model_spec(ssp_w, v, sigma, sigmaK, delta, kappa, mpolyweights, FZqmu, u, nfft, xpoly, losvd)
    # Masking with numpyro.handlers.mask doesn't work - chains don't move
    # with numpyro.handlers.mask(mask=mask):
    #     y = numpyro.sample('y', dist.Normal(ybar, lerr), obs=lspec)
    # ... use the mask manually instead
    y = numpyro.sample('y', dist.Normal(ybar[mask], lerr[mask]))
    evaluated_losvd = numpyro.deterministic(
        'losvd',
        losvd.evaluate_via_fft(v, sigma, sigmaK, delta, kappa, vmax=2000, nv=1001)
        )
    return    

def fit_spectrum(q_pca, mdegree):
    try:
        with open(f'../data/spectrum.out', 'rb') as f:
            args = pickle.load(f)
    except FileNotFoundError:
        return None
    lspec, lerr, FZqmu, u, cube, xpoly, mask, A_eq, b_eq = args
    data = {'lspec':lspec,
            'lerr':lerr,
            'FZqmu':FZqmu,
            'u':u,
            'nfft':cube.ssps.n_fft,
            'xpoly':xpoly,
            'mask':np.load('../data/emission_line_mask.npy'),
            'q_pca':q_pca,
            'mdegree':mdegree,
            'losvd':losvd
    }
    kernel = NUTS(
        numpyro_model,
        target_accept_prob=0.75,
        )
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=3)
    mcmc.run(PRNGKey(4674343), extra_fields=("potential_energy",), **data)
    mcmc.print_summary()
    predictive = Predictive(
        numpyro_model_for_predictive_distribution,
        posterior_samples=mcmc.get_samples()
    )
    posterior_predictive_samples = predictive(PRNGKey(54356), **data)
    return mcmc, data, posterior_predictive_samples

if __name__ == "__main__":
    q_pca = 20
    mdegree = 5
    mcmc, data, posterior_predictive_samples = fit_spectrum(q_pca, mdegree)
    result = {
        'q_pca':q_pca,
        'mdegree':mdegree,
        'mcmc':mcmc,
        'data':data,
        'posterior_predictive':posterior_predictive_samples
        }
    with open(f'../output/mcmc_result.out', 'wb') as outfile:
        dill.dump(result, outfile)