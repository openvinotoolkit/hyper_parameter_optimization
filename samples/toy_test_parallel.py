# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Parallel version of toy test example with skopt.benchmarks.branin
# https://scikit-optimize.github.io/stable/modules/generated/skopt.benchmarks.branin.html
# 
# Three minima points
# (9.42478, 2.475)
# (3.141592, 2.275)
# (-3.141592, 12.275)
#

import hpopt

import numpy as np
from skopt.benchmarks import branin, hart6

from multiprocessing import Process

def my_model(a, b):
    return -branin((a, b))

def my_trainer(config):
    for iteration_num in range(config["iterations"]):
        score = my_model(**config["params"])
        if hpopt.report(config=config, score=score) == hpopt.Status.STOP:
            break

hp_configs = {"a": hpopt.search_space("uniform", [-5, 10]),
              "b": hpopt.search_space("uniform", [0, 15])}

my_hpo = hpopt.create(save_path='./tmp/my_hpo_branin',
                      search_alg="bayes_opt",
                      search_space=hp_configs,
                      ealry_stop="median_stop",
                      num_init_trials=5, num_trials=50, max_iterations=2,
                      num_full_iterations=2, full_dataset_size=1)

while True:
    configs = my_hpo.get_next_samples()

    if len(configs) == 0:
        break

    proc_list = []
    for config in configs:
        p = Process(target=my_trainer, args=(config, ))
        proc_list.append(p)
        p.start()

    for p in proc_list:
        p.join()

print("best hp: ", my_hpo.get_best_config())
