import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

#TODO: Implement this under log to prevent underflow

# Parameters
BASE_MEAN = -3
BASE_VAR = 0.1

MODE1_MEAN = (-1, -1)
MODE1_VAR = (0.1, 0.1)
MODE2_MEAN = (1, 1)
MODE2_VAR = (0.05, 0.05)

TOTAL_DISTRIBUTIONS = 200
TOTAL_RUNS = 1000

np.random.seed(42069)

def calculate_f0_(state, mean, var):
    """Calculate f(state), where f is proportional to the true P (normal in this case)"""
    coeff = 1 / (2 * np.pi * var ** 2) ** 3
    func = np.exp(-0.5 * np.sum((state - mean)**2 / var**2))
    return coeff * func
    #f_0 = 0.5 * np.sum(((state - mean) ** 2) / (0.1 ** 2))
    #coeff = (2 * np.pi * 0.1 ** 2) ** (6 / 3)
    #return coeff * f_0

def calculate_f0(state):
    """Calculate f0, ie. the distribution of interest. f0 = N(BASE_MEAN, BASE_VAR)"""
    return calculate_f0_(state, BASE_MEAN, BASE_VAR)

def calculate_f0_mixture(state):
    """Calculate f0, ie. the distribution of interest. This is the bimodal Gaussian mixture version"""
    component1 = np.exp(np.sum((state - 1) ** 2 / (0.1 ** 2)) * -0.5)
    component2 = 128 * np.exp(np.sum((state + 1) ** 2 / (0.05 ** 2)) * -0.5) 
    norm_const = 1 / (2 * np.pi * 0.1 ** 2) ** 3
    return (component1 + component2) #norm_const * (component1 + component2)

def calculate_fn(state):
    """Calculate fn, ie. the simple distribution. fn = N(0, 1)"""
    return calculate_f0_(state, 0, 1)

def calculate_fj(state, beta_schedule, j):
    """Calculate an intermediate distribution function fj"""
    return (calculate_f0_mixture(state) ** beta_schedule[j]) * (calculate_fn(state) ** (1 - beta_schedule[j])) 

def log_mh_comparison(state, candidate_state, beta_schedule, j):
    f0 = calculate_f0_mixture(state.astype(np.longdouble))
    fn = calculate_fn(state.astype(np.longdouble))
    f0_p = calculate_f0_mixture(candidate_state.astype(np.longdouble))
    fn_p = calculate_fn(candidate_state.astype(np.longdouble))
    log_prime = beta_schedule[j] * np.log(f0_p) + (1 - beta_schedule[j]) * np.log(fn_p)
    log_current = beta_schedule[j] * np.log(f0) + (1 - beta_schedule[j]) * np.log(fn)
    log_accept = log_prime - log_current
    return np.exp(log_accept)

def mh_update(current_state, cov, beta_schedule, j):
    """Metropolis-Hastings update"""
    mean = current_state
    cov = cov * np.identity(current_state.shape[0])
    candidate_state = random.multivariate_normal(mean=mean, cov=cov)
    acceptance_ratio = log_mh_comparison(current_state, candidate_state, beta_schedule, j)

    """
    fx_prime = calculate_fj(candidate_state, beta_schedule, j)
    fx_current = calculate_fj(current_state, beta_schedule, j)
    if fx_prime == 0 and fx_current == 0:
        breakpoint()
    acceptance_ratio = np.divide(fx_prime, fx_current, np.zeros_like(fx_prime), where=fx_current!=0)
    """
    
    u = random.uniform()
    new_state = candidate_state if u <= acceptance_ratio else current_state
    return new_state
#mh_update(np.ones(6), np.identity(6), [0.1], 0)

def sample_transition(state, beta_schedule, j):
    """Transitions implemented by Metropolis-Hastings updates"""
    covariances = [0.05**2, 0.15**2, 0.5**2]
    new_state = state
    for _ in range(10):
        for cov in covariances:
            new_state = mh_update(new_state, cov, beta_schedule, j)    # Note that paper uses multiple proposal distributions
    return new_state

def log_weight(state, beta_schedule, j):
    f0 = calculate_f0_mixture(state.astype(np.longdouble))
    fn = calculate_fn(state.astype(np.longdouble))
    f0_next = calculate_f0_mixture(state.astype(np.longdouble))
    fn_next = calculate_fn(state.astype(np.longdouble))
    f = np.log(f0) * beta_schedule[j] + np.log(fn) * (1 - beta_schedule[j])
    f_next = np.log(f0) * beta_schedule[j - 1] + np.log(fn) * (1 - beta_schedule[j - 1])
    return f_next - f

def annealing_run(beta_schedule):
    mean = np.zeros(6)
    cov = np.identity(6)
    current_state = random.multivariate_normal(mean=mean, cov=cov)

    weights = []
    states = []
    for j in range(TOTAL_DISTRIBUTIONS - 1, 0, -1):
        current_state = sample_transition(current_state, beta_schedule, j) 
        log_w = log_weight(current_state, beta_schedule, j)

        weights.append(log_w)

        states.append(current_state)
    weights = np.array(weights)
    states = np.array(states)

    #plt.scatter(np.flip(beta_schedule[1:]), states[:, 0])
    #plt.show()
    #breakpoint()
    
    final_log_w = np.add.reduce(weights)
    final_state = current_state
    return final_log_w, final_state, weights, states 

def calculate_final_mean(weights, states):
    return (np.expand_dims(np.exp(weights), 1) * states).sum(0) / np.exp(weights).sum(0)

def sample_ais(beta_schedule):
    weights = []
    states = []
    all_weights = []
    all_states = []
    from tqdm import tqdm
    for _ in tqdm(range(TOTAL_RUNS)):
        log_w, state, all_weight, all_state = annealing_run(beta_schedule)
        weights.append(log_w)
        states.append(state)
        all_weights.append(all_weight)
        all_states.append(all_state)
    all_weights = np.array(all_weights)
    all_states = np.array(all_states)
    fig = plt.figure()
    plt.scatter(np.tile(np.flip(beta_schedule[0::20]), TOTAL_RUNS), all_states[:, 0::20, 0].flatten(), s=1)
    #plt.scatter(np.tile(np.flip(beta_schedule[1:]), TOTAL_RUNS), all_states[:, :, 0].flatten(), s=1)
    plt.xlabel("beta")
    plt.ylabel("first component of state")
    plt.show()
    breakpoint()
    return calculate_final_mean(np.array(weights), np.array(states)), np.exp(np.array(weights)), np.array(states)


beta_schedule1 = np.linspace(0.01, 0, num=40, endpoint=True)
beta_schedule2 = np.geomspace(1, 0.01, num=160, endpoint=False)
beta_schedule = np.concatenate((beta_schedule2, beta_schedule1))

mean, weights, states = sample_ais(beta_schedule)
print(mean)
breakpoint()
