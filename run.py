import GPyOpt
from experiments import experiment_fn
import numpy as np
import matplotlib.pyplot as plt

MAX_TIME = 600
BOUNDS = [
    {
        'name': 'reg_lambda_dist',
        'type': 'continuous',
        'domain': (0.0005, 0.005)
    },
    {
        'name': 'reg_lambda_w',
        'type': 'continuous',
        'domain': (0.005, 0.05)
    },
    {
        'name': 'reg_lambda_p',
        'type': 'continuous',
        'domain': (0.00005, 0.0005)
    },
    {
        'name': 'lr_prot',
        'type': 'continuous',
        'domain': (0.00001, 0.0001)
    },
    {
        'name': 'lr_weights',
        'type': 'continuous',
        'domain': (0.00001, 0.0001)
    },
    {
        'name': 'reg_w',
        'type': 'discrete',
        'domain': (1, 2)
    },
    {
        'name': 'n_prototypes',
        'type': 'discrete',
        'domain': (2, 6)
    }  # will be x2. ie if 2, then number of prototypes will actually be 4, if 4 then 8, etc.
]


def run_for_dataset(dataset):
  np.random.seed(777)
  optimizer = GPyOpt.methods.BayesianOptimization(f=experiment_fn,
                                                  domain=BOUNDS,
                                                  acquisition_type='MPI',
                                                  acquisition_par=0.3,
                                                  exact_eval=True,
                                                  maximize=True)
  max_iter = 2
  optimizer.run_optimization(max_iter, max_time=600)
  optimizer.plot_convergence(filename="optimizer_bayesopt.png")
  print(optimizer.Y_best[-1])
  print(optimizer.x_opt)


run_for_dataset("Elephant")
