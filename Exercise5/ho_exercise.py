import matplotlib
matplotlib.use('Agg')
import sys
import pickle
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from copy import deepcopy
from robo.fmin import bayesian_optimization

rf = pickle.load(open("./rf_surrogate_cnn.pkl", "rb"))
cost_rf = pickle.load(open("./rf_cost_surrogate_cnn.pkl", "rb"))
lower = [-6, 32, 4, 4, 4]
upper = [0, 512, 10, 10, 10]
runs = 10
iterations = 50

  
def get_random_hyperparameters_list():
    """
        Get random list of hyperparameters
    """
    return [np.random.uniform(l, u) for l, u in zip(lower, upper)]
    

def objective_function(x, epoch=40):
    """
        Function wrapper to approximate the validation error of the hyperparameter configurations x by the prediction of a surrogate regression model,
        which was trained on the validation error of randomly sampled hyperparameter configurations.
        The original surrogate predicts the validation error after a given epoch. Since all hyperparameter configurations were trained for a total amount of 
        40 epochs, we will query the performance after epoch 40.
    """
    
    # Normalize all hyperparameter to be in [0, 1]
    x_norm = deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)
    

    x_norm = np.append(x_norm, epoch)
    y = rf.predict(x_norm[None, :])[0]

    return y

def runtime(x, epoch=40):
    """
        Function wrapper to approximate the runtime of the hyperparameter configurations x.
    """
    
    # Normalize all hyperparameter to be in [0, 1]
    x_norm = deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)
    

    x_norm = np.append(x_norm, epoch)
    y = cost_rf.predict(x_norm[None, :])[0]

    return y

def random_search():
    """
        Random search implementation
    """
    optimal_value = sys.maxsize
    performance_values = np.zeros(shape=(runs, iterations))
    runtime_values = np.zeros(shape=(runs, iterations))

    for i in range(runs):
        for j in range(iterations):
            hps = get_random_hyperparameters_list()
            output_value = objective_function(hps)
            runtime_values[i, j] = runtime(hps)

            if output_value < optimal_value:
                performance_values[i,j] = output_value
                optimal_value = output_value
            else:
                performance_values[i,j] = optimal_value

    performance_mean = np.mean(performance_values, axis=0)
    runtime_mean = np.cumsum(np.mean(runtime_values, axis=0), axis=0)
    return performance_mean, runtime_mean

def bayesian_optimization():
    """
        Bayesian optimization implementation
    """
    performance_values = list()
    runtime_values = list()
    for i in range(runs):
        output_value = bayesian_optimization(objective_function, lower, upper, num_iterations=iterations)
        performance_values.append(output_value['incumbent_values'])
        runtime_values.append(output_value['runtime'])

    performance_mean = np.mean(performance_values, axis=0)
    runtime_mean = np.cumsum(np.mean(runtime_values, axis=0), axis=0)
    return performance_mean, runtime_mean
    


def plot_and_save(rs, bo, xlabel_value, ylabel_value, path_to_png):
    """
        plot and save the figure
    """
    plt.plot(range(1,len(rs) + 1), rs, color="green", linewidth=2.5, linestyle="-", label="Random Search")
    plt.plot(range(1,len(bo) + 1), bo, color="blue", linewidth=2.5, linestyle="-", label="Bayesian Optimization")
    plt.legend(loc='upper right', frameon=False)
    plt.xlabel(xlabel_value)
    plt.ylabel(ylabel_value)
    plt.savefig(path_to_png)
    plt.show()
    
def main():
    rs_performance, rs_runtime = random_search()
    bo_performance, bo_runtime = bayesian_optimization()
    plot_and_save(rs_performance, bo_performance, "Performance", "Iterations", "performance.png")
    plot_and_save(rs_runtime, bo_runtime, "Runtime", "Iterations", "performance.png")
    
if __name__ == "__main__":
    main()
