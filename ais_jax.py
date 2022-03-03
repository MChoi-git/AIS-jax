from functools import partial
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as random
# Extra precision is needed 
from jax.config import config; config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

@jax.jit
def normal_density(state: jnp.ndarray, mean: int, var: int) -> int:
    """Calculate probability of multivariate Gaussian
    Args:
        state (jnp.ndarray): Point at which the probability is calculated
        mean (int): Mean of Gaussian
        var (int): Variance of Gaussian
    """
    return (1 / (2 * jnp.pi * var ** 2) ** 3) * jnp.exp(-0.5 * jnp.sum((state - mean)**2 / var**2))

fn = partial(normal_density, mean=0, var=1) # Fixing a standard normal

@jax.jit
def f0(state: jnp.ndarray) -> int:
    """Calculate probability of target multivariate mixture of 2 Gaussians
    Args:
        state (jnp.ndarray): Point at which the probability is calculated
    """
    component1 = jnp.exp(jnp.sum((state - 1) ** 2 / (0.1 ** 2)) * -0.5)
    component2 = 128 * jnp.exp(jnp.sum((state + 1) ** 2 / (0.05 ** 2)) * -0.5)
    norm_const = 1 / (2 * jnp.pi * 0.1 ** 2) ** 3
    return norm_const * (component1 + component2)

@jax.jit
def log_acceptance(state: jnp.ndarray, candidate_state: jnp.ndarray, beta: float) -> jnp.ndarray:
    """Calculate the log of the acceptance rate
    Args:
        state (jnp.ndarray): Current state
        candidate_state (jnp.ndarray): Next potential state
        beta (float): Current beta value
    """
    log_fprime = jnp.log(f0(candidate_state)) * beta + jnp.log(fn(candidate_state)) * (1 - beta)
    log_f = jnp.log(f0(state)) * beta + jnp.log(fn(state)) * (1 - beta)
    return log_fprime - log_f

@jax.jit
def mh_update(state: jnp.ndarray, cov_subkey_pair: Tuple[jnp.ndarray, jnp.ndarray], beta: float):
    """Do one Metropolis-Hastings update
    Args:
        state (jnp.ndarray): Current state
        cov_subkey_pair (tuple): Covariance matrix to sample the next state and the rng subkey
        beta (float): Current beta value
    """
    cov, subkey = cov_subkey_pair
    key1, key2 = random.split(subkey)
    candidate_state = random.multivariate_normal(key1, state, cov)
    a = jnp.exp(log_acceptance(state, candidate_state, beta))
    u = random.uniform(key2)
    next_state = jnp.where(u <= a, candidate_state, state)
    return next_state, next_state
#mh_update(random.multivariate_normal(random.PRNGKey(2), jnp.zeros(6), jnp.identity(6)), (jnp.identity(6) * 0.015**2, random.PRNGKey(1)), 0.1)

@jax.jit
def mh_transitions(state: jnp.ndarray, betas_subkeys: Tuple[float, jnp.ndarray], cov: jnp.ndarray):
    """Do a series of Metropolis-Hastings updates
    Args:
        state (jnp.ndarray): Starting state
        betas_subkeys (tuple): Current beta value and rng subkey
        cov (jnp.ndarray): Covariances to do Metropolis-Hastings updates against
    """
    beta, subkey = betas_subkeys
    update_keys = random.split(subkey, 10 * 3)
    covariances = jax.vmap(lambda x: x * jnp.identity(6))(cov).tile((10, 1, 1))
    mh_update_ = partial(mh_update, beta=beta)
    final_state, _ = jax.lax.scan(mh_update_, state, (covariances, update_keys)) 
    return final_state, final_state 
#mh_transitions(random.multivariate_normal(random.PRNGKey(2), jnp.zeros(6), jnp.identity(6)), (0.1, random.PRNGKey(1)), cov = jnp.array([0.01**2, 0.15**2, 0.5**2]))

@jax.jit
def annealing_run(subkey: jnp.ndarray, betas: jnp.ndarray):
    """Do one full annealing run to produce one sample x0
    Args:   
        subkey (jnp.ndarray): Rng subkey
        betas (jnp.ndarray): Full beta schedule
    """
    init_key, *transition_keys = random.split(subkey, 200 + 1)  # 200 distributions
    transition_keys = jnp.array(transition_keys)
    cov = jnp.array([0.05**2, 0.15**2, 0.5**2])
    init_state = random.multivariate_normal(init_key, jnp.zeros(6), jnp.identity(6))
    final_state, all_final_states = jax.lax.scan(partial(mh_transitions, cov=cov), init_state, (betas, transition_keys))
    return final_state, all_final_states
#annealing_run(random.PRNGKey(1), jnp.concatenate((jnp.linspace(0, 0.01, num=40, endpoint=True), jnp.geomspace(0.01, 1, num=160, endpoint=False))))

def ais():
    """Main function to sample 1000 x0 using the same number of annealing runs. 
    TODO: 
        - Implement importance weight calculation
    Args:   
    """
    head_key = random.PRNGKey(42069)
    run_keys = random.split(head_key, num=1000)
    betas1 = jnp.linspace(0, 0.01, num=40, endpoint=True)
    betas2 = jnp.geomspace(0.01, 1, num=160, endpoint=False)
    betas = jnp.concatenate((betas1, betas2))
    final_states, all_states = jax.vmap(partial(annealing_run, betas=betas))(run_keys)
    #fig = plt.figure()
    #plt.scatter(jnp.tile(betas[0::20], 1000), all_states[:, 0::20, 0].flatten(), s=1)
    #plt.show()
    #breakpoint()
    return final_states
    

start_time = time.time()
ais()
end_time = time.time()
print(f"time elapsed: {end_time - start_time}")
