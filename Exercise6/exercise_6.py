import numpy as np
import pickle
import argparse
import ConfigSpace as CS
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.ERROR)
from copy import deepcopy

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario

import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker


def create_config_space():
    cs = CS.ConfigurationSpace()

    Adam_final_lr_fraction = CS.UniformFloatHyperparameter('Adam_final_lr_fraction',
                             lower=1e-4,
                             upper=1,
                             default_value=1e-2,
                             log=True)

    Adam_initial_lr = CS.UniformFloatHyperparameter('Adam_initial_lr',
                      lower=1e-4,
                      upper=1e-2,
                      default_value=1e-3,
                      log=True)

    SGD_final_lr_fraction = CS.UniformFloatHyperparameter('SGD_final_lr_fraction',
                      lower=1e-4,
                      upper=1,
                      default_value=1e-2,
                      log=True)

    SGD_initial_lr = CS.UniformFloatHyperparameter('SGD_initial_lr',
                      lower=1e-3,
                      upper=0.5,
                      default_value=1e-1,
                      log=True)

    SGD_momentum = CS.UniformFloatHyperparameter('SGD_momentum',
                      lower=0.0,
                      upper=0.99,
                      default_value=0.9,
                      log=False)

    StepDecay_epochs_per_step = CS.UniformIntegerHyperparameter("StepDecay_epochs_per_step",
                                               lower=1,
                                               upper=128,
                                               default_value=16,
                                               log=False)

    activation = CS.CategoricalHyperparameter("activation",
                                              ["relu", "tanh"],
                                              default_value="relu")

    batch_size = CS.UniformIntegerHyperparameter("batch_size",
                                               lower=8,
                                               upper=256,
                                               default_value=16,
                                               log = True)

    dropout_0 = CS.UniformFloatHyperparameter('dropout_0',
                      lower=0.0,
                      upper=0.5,
                      default_value=0.0,
                      log=False)

    dropout_1 = CS.UniformFloatHyperparameter('dropout_1',
                      lower=0.0,
                      upper=0.5,
                      default_value=0.0,
                      log=False)

    dropout_2 = CS.UniformFloatHyperparameter('dropout_2',
                      lower=0.0,
                      upper=0.5,
                      default_value=0.0,
                      log=False)

    dropout_3 = CS.UniformFloatHyperparameter('dropout_3',
                      lower=0.0,
                      upper=0.5,
                      default_value=0.0,
                      log=False)

    l2_reg_0 = CS.UniformFloatHyperparameter('l2_reg_0',
                      lower=1e-6,
                      upper=1e-2,
                      default_value=1e-4,
                      log=True)

    l2_reg_1 = CS.UniformFloatHyperparameter('l2_reg_1',
                      lower=1e-6,
                      upper=1e-2,
                      default_value=1e-4,
                      log=True)

    l2_reg_2 = CS.UniformFloatHyperparameter('l2_reg_2',
                      lower=1e-6,
                      upper=1e-2,
                      default_value=1e-4,
                      log=True)

    l2_reg_3 = CS.UniformFloatHyperparameter('l2_reg_3',
                      lower=1e-6,
                      upper=1e-2,
                      default_value=1e-4,
                      log=True)

    learning_rate_schedule = CS.CategoricalHyperparameter("learning_rate_schedule",
                                                          ['ExponentialDecay', 'StepDecay'],
                                                          default_value="ExponentialDecay")

    loss_function = CS.CategoricalHyperparameter("loss_function",
                                                 ['categorical_crossentropy'],
                                                 default_value="categorical_crossentropy")

    num_layers = CS.UniformIntegerHyperparameter("num_layers",
                                               lower=1,
                                               upper=4,
                                               default_value=2,
                                               log=False
                                               )

    num_units_0 = CS.UniformIntegerHyperparameter("num_units_0",
                                               lower=16,
                                               upper=256,
                                               default_value=32,
                                               log=True)

    num_units_1 = CS.UniformIntegerHyperparameter("num_units_1",
                                               lower=16,
                                               upper=256,
                                               default_value=32,
                                               log=True)

    num_units_2 = CS.UniformIntegerHyperparameter("num_units_2",
                                               lower=16,
                                               upper=256,
                                               default_value=32,
                                               log=True)

    num_units_3 = CS.UniformIntegerHyperparameter("num_units_3",
                                               lower=16,
                                               upper=256,
                                               default_value=32,
                                               log=True)

    optimizer = CS.CategoricalHyperparameter("optimizer",
                                             ["Adam", "SGD"],
                                             default_value = 'Adam')

    output_activation = CS.CategoricalHyperparameter("output_activation",
                                                     ["softmax"],
                                                     default_value = "softmax")

    #Adding hyperparameters to config space
    cs.add_hyperparameter(Adam_final_lr_fraction)
    cs.add_hyperparameter(Adam_initial_lr)
    cs.add_hyperparameter(SGD_final_lr_fraction)
    cs.add_hyperparameter(SGD_initial_lr)
    cs.add_hyperparameter(SGD_momentum)
    cs.add_hyperparameter(StepDecay_epochs_per_step)
    cs.add_hyperparameter(activation)
    cs.add_hyperparameter(batch_size)
    cs.add_hyperparameter(dropout_0)
    cs.add_hyperparameter(dropout_1)
    cs.add_hyperparameter(dropout_2)
    cs.add_hyperparameter(dropout_3)
    cs.add_hyperparameter(l2_reg_0)
    cs.add_hyperparameter(l2_reg_1)
    cs.add_hyperparameter(l2_reg_2)
    cs.add_hyperparameter(l2_reg_3)
    cs.add_hyperparameter(learning_rate_schedule)
    cs.add_hyperparameter(loss_function)
    cs.add_hyperparameter(num_layers)
    cs.add_hyperparameter(num_units_0)
    cs.add_hyperparameter(num_units_1)
    cs.add_hyperparameter(num_units_2)
    cs.add_hyperparameter(num_units_3)
    cs.add_hyperparameter(optimizer)
    cs.add_hyperparameter(output_activation)

    #conditions
    cond = CS.EqualsCondition(Adam_final_lr_fraction, optimizer, 'Adam')
    cs.add_condition(cond)
    cond = CS.EqualsCondition(Adam_initial_lr, optimizer, 'Adam')
    cs.add_condition(cond)
    cond = CS.EqualsCondition(SGD_final_lr_fraction, optimizer, 'SGD')
    cs.add_condition(cond)
    cond = CS.EqualsCondition(SGD_initial_lr, optimizer, 'SGD')
    cs.add_condition(cond)
    cond = CS.EqualsCondition(SGD_momentum, optimizer, 'SGD')
    cs.add_condition(cond)
    cond = CS.EqualsCondition(StepDecay_epochs_per_step, learning_rate_schedule, 'StepDecay')
    cs.add_condition(cond)
    cond = CS.GreaterThanCondition(dropout_1, num_layers, 2)
    cs.add_condition(cond)
    cond = CS.GreaterThanCondition(dropout_2, num_layers, 3)
    cs.add_condition(cond)
    cond = CS.EqualsCondition(dropout_3, num_layers, 4)
    cs.add_condition(cond)
    cond = CS.GreaterThanCondition(l2_reg_1, num_layers, 2)
    cs.add_condition(cond)
    cond = CS.GreaterThanCondition(l2_reg_2, num_layers, 3)
    cs.add_condition(cond)
    cond = CS.EqualsCondition(l2_reg_3, num_layers, 4)
    cs.add_condition(cond)
    cond = CS.GreaterThanCondition(num_units_1, num_layers, 2)
    cs.add_condition(cond)
    cond = CS.GreaterThanCondition(num_units_2, num_layers, 3)
    cs.add_condition(cond)
    cond = CS.EqualsCondition(num_units_3, num_layers, 4)
    cs.add_condition(cond)

    return cs


def objective_function(config, epoch=127, **kwargs):
    # Cast the config to an array such that it can be forwarded to the surrogate
    x = deepcopy(config.get_array())
    x[np.isnan(x)] = -1
    lc = rf.predict(x[None, :])[0]
    c = cost_rf.predict(x[None, :])[0]

    return lc[epoch], {"cost": c, "learning_curve": lc[:epoch].tolist()}


class WorkerWrapper(Worker):
    def compute(self, config, budget, *args, **kwargs):
        cfg = CS.Configuration(cs, values=config)
        loss, info = objective_function(cfg, epoch=int(budget))

        return ({
            'loss': loss,
            'info': {"runtime": info["cost"],
                     "lc": info["learning_curve"]}
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_smac', action='store_true')
    parser.add_argument('--run_hyperband', action='store_true')
    parser.add_argument('--n_iters', default=50, type=int)
    args = vars(parser.parse_args())

    n_iters = args['n_iters']

    cs = create_config_space()
    rf = pickle.load(open("./rf_surrogate_paramnet_mnist.pkl", "rb"))
    cost_rf = pickle.load(open("./rf_cost_surrogate_paramnet_mnist.pkl", "rb"))

    if args["run_smac"]:
        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": n_iters,
                             "cs": cs,
                             "deterministic": "true",
                             "output_dir": ""})

        smac = SMAC(scenario=scenario, tae_runner=objective_function)
        smac.optimize()

        # The following lines extract the incumbent strategy and the estimated wall-clock time of the optimization
        rh = smac.runhistory
        incumbents = []
        incumbent_performance = []
        inc = None
        inc_value = 1
        idx = 1
        t = smac.get_trajectory()

        wall_clock_time = []
        cum_time = 0
        for d in rh.data:
            cum_time += rh.data[d].additional_info["cost"]
            wall_clock_time.append(cum_time)
        for i in range(n_iters):

            if idx < len(t) and i == t[idx].ta_runs - 1:
                inc = t[idx].incumbent
                inc_value = t[idx].train_perf
                idx += 1

            incumbents.append(inc)
            incumbent_performance.append(inc_value)

        f, (ax1, ax2) = plt.subplots(nrows=2, sharey=False)
        ax1.set_xlabel("Wall Clock Time")
        ax1.set_ylabel("Incumbent Performance")
        ax1.plot(wall_clock_time, incumbent_performance)
        ax1.set_title('SMAC')

        lc_smac = []
        for i, d in enumerate(rh.data):
            lc_smac.append(rh.data[d].additional_info["learning_curve"])


        ax2.plot(lc_smac)
        ax2.set_xlabel("No of Iterations")
        ax2.set_ylabel("Error")
        ax2.set_title('SMAC learning curves')
        plt.tight_layout()
        f.savefig('SMAC.png')
        plt.show()

    if args["run_hyperband"]:
        nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()

        # starting the worker in a separate thread
        w = WorkerWrapper(nameserver=nameserver, ns_port=ns_port)
        w.run(background=True)

        CG = hpbandster.config_generators.RandomSampling(cs)

        # instantiating Hyperband with some minimal configuration
        HB = hpbandster.HB_master.HpBandSter(
            config_generator=CG,
            run_id='0',
            eta=2,  # defines downsampling rate
            min_budget=1,  # minimum number of epochs / minimum budget
            max_budget=127,  # maximum number of epochs / maximum budget
            nameserver=nameserver,
            ns_port=ns_port,
            job_queue_sizes=(0, 1),
        )
        # runs one iteration if at least one worker is available
        res = HB.run(10, min_n_workers=1)

        # shutdown the worker and the dispatcher
        HB.shutdown(shutdown_workers=True)

        # extract incumbent trajectory and all evaluated learning curves
        traj = res.get_incumbent_trajectory()
        wall_clock_time = []
        cum_time = 0

        for c in traj["config_ids"]:
            cum_time += res.get_runs_by_id(c)[-1]["info"]["runtime"]
            wall_clock_time.append(cum_time)


        incumbent_performance = traj["losses"]

        f, (ax1, ax2) = plt.subplots(nrows=2, sharey=False)
        ax1.set_xlabel("Wall Clock Time")
        ax1.set_ylabel("Incumbent Performance")
        ax1.plot(wall_clock_time, incumbent_performance)
        ax1.set_title('Hyperband')
        # TODO: save and plot the wall clock time and the validation of the incumbent after each iteration here

        lc_hyperband = []
        for i, r in enumerate(res.get_all_runs()):
            c = r["config_id"]
            lc_hyperband.append(res.get_runs_by_id(c)[-1]["info"]["lc"])

            ax2.plot(range(len(res.get_runs_by_id(c)[-1]["info"]["lc"])), lc_hyperband[i])
            ax2.set_xlabel("No of Iterations")
            ax2.set_ylabel("Error")
            ax2.set_title('Hyperband learning curves')

        plt.tight_layout()
        f.savefig('Hyperband.png')
        plt.show()
