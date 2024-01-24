import numpy as np
from pyswarm import pso

def objective_function(x):
    """
    Objective function for optimization.
    Rounds the first element of x and computes the objective value.
    """
    x[0] = np.round(x[0], 0)
    return -(x[0] + x[1] * x[0])

def constraints(x):
    """
    Constraint function for optimization.
    Returns a penalty if constraints are not met.
    """
    penalty = 0
    if not -x[0] + 2 * x[1] * x[0] <= 8: 
        penalty = np.inf
    if not 2 * x[0] + x[1] <= 14: 
        penalty = np.inf
    if not 2 * x[0] - x[1] <= 10: 
        penalty = np.inf
    return penalty

# Define bounds and initial guess
lower_bound = [3, 6]
upper_bound = [-3, 6]
initial_guess = [0, 0]

try:
    optimal_solution, optimal_objective_value = pso(objective_function, lower_bound, upper_bound, initial_guess, constraints)

    print('Optimal x =', optimal_solution[0])  # Best value of x
    print('Optimal y =', optimal_solution[1])  # Best value of y
except Exception as e:
    print(f"An error occurred during optimization: {e}")
