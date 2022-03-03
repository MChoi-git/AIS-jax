import os
from functools import partial

import numpy as np
import numpy.random as random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

TRUE_MEAN = (3, -3)
TRUE_VAR = (1, 1)
ITERATIONS = 1000
NUM_WALKS = 5

np.random.seed(42069)

def sample_gx(state: np.ndarray):
    """Given 2d state sample g(x|y) = N(mean(y), 1), y=state"""
    return random.randn(2) + state

def calculate_f_(state: np.ndarray, mean: np.ndarray, var: np.ndarray):
    """Calculate the probability of the given state under normal distribution of given mean and var"""
    coeff = 1 / (np.sqrt(var) * np.sqrt(2 * np.pi))
    exp = np.exp(-0.5 * ((state - mean) / var) ** 2)
    return coeff * exp

calculate_f = partial(calculate_f_, mean=TRUE_MEAN, var=TRUE_VAR)

def gen_next_state(current_state: np.ndarray):
    """Metropolis-Hastings update returning the next state"""
    candidate_state = sample_gx(current_state)

    fx_prime = calculate_f(candidate_state)
    fx_current = calculate_f(current_state)
    acceptance_ratio = fx_prime / fx_current

    u = random.uniform(2)
    current_state = np.where(acceptance_ratio >= u, candidate_state, current_state)
    return current_state

def mcmc(total_iters: int, current_state: np.ndarray):
    """MCMC iteration loop, returning the entire walk history"""
    samples = [current_state]
    for t in range(total_iters):
        current_state = gen_next_state(current_state)
        samples.append(current_state)
    return np.array(samples)

def update_line(num, walks, lines):
    """Update the plot line data"""
    for walk, line in zip(walks, lines):
        line.set_data(walk[:num, :2].T)
    return lines

# Walk
init_state = random.randn(2)
walks = [mcmc(ITERATIONS, init_state) for _ in range(NUM_WALKS)]    # (ITERATIONS, 2)

# Plot
fig = plt.figure(figsize=(4, 4), dpi=100)
ax = fig.add_subplot(aspect='equal', xlim=(-5, 5), ylim=(-5, 5))
ax.set_title(f"Metropolis-Hastings (n={ITERATIONS}) MCMC - Gaussian2d ({TRUE_MEAN}, {TRUE_VAR})")
lines = [ax.plot([], [])[0] for _ in range(NUM_WALKS)]
ax.set(xlabel="X")
ax.set(ylabel="Y")
ani = FuncAnimation(
    fig, update_line, ITERATIONS, fargs=(walks, lines), interval=10, repeat=True)  
#ani.save('metropolis_hastings_mcmc.gif', fps=30)
plt.show()
