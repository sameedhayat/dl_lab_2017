import pickle
import numpy as np
import sys
from copy import deepcopy
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

rf = pickle.load(open("./rf_surrogate_cnn.pkl", "rb"))
cost_rf = pickle.load(open("./rf_cost_surrogate_cnn.pkl", "rb"))
learning_rate = [-6, 0]
batch_size = [32, 512]
filter_layer1 = [4, 10]
filter_layer2 = [4, 10]
filter_layer3 = [4, 10]

def get_random_from_range(x):
    """
        Get random number between the lower and upper bound
    """
    return np.random.uniform(x[0], x[1])
    
def get_random_hyperparameters_list():
    """
        Get random list of hyperparameters
    """
    ret = list()
    ret.append(get_random_from_range(learning_rate))
    ret.append(get_random_from_range(batch_size))
    ret.append(get_random_from_range(filter_layer1))
    ret.append(get_random_from_range(filter_layer2))
    ret.append(get_random_from_range(filter_layer3))
    return ret
    
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
    runs = 10
    iterations = 50
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
            
def main():
    performance_mean, runtime_mean = random_search()
    plt.plot(range(1,len(performance_mean) + 1), performance_mean)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.xlabel('number of iterations')
    plt.ylabel('training loss')
    plt.savefig('loss.png')
    plt.show()
    
if __name__ == "__main__":
    main()
