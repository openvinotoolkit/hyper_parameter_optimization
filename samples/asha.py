import hpopt
import numpy as np
from multiprocessing import Process
import time

def my_model(step, width, height):
    return (0.1 + width * step / 100)**(-1) + height * 0.1


def my_trainer(config):
    # Hyperparameters
    width, height = config['params']['width'], config['params']['height']

    for step in range(config["iterations"]):
        # Iterative training function - can be an arbitrary training procedure
        score = my_model(step, width, height)
        # 
        if hpopt.report(config=config, score=score) == hpopt.Status.STOP:
            break


hp_configs = {"width": hpopt.search_space("uniform", [10, 100]),
              "height": hpopt.search_space("uniform", [0, 100])}

my_hpo = hpopt.create(save_path='./tmp/my_hpo_asha',
                      search_alg="asha",
                      search_space=hp_configs,
                      mode='min',
                      num_trials=100,
                      min_iterations=1,
                      max_iterations=100,
                      num_brackets=5,
                      reduction_factor=3,
                      num_full_iterations=50,
                      full_dataset_size=2500)
print(my_hpo.rungs_in_brackets)
exit()
num_max_workers = 10
proc_list = []

while True:
    num_active_workers = 0
    for p in proc_list:
        if p.is_alive():
            num_active_workers += 1
        else:
            p.close()
            proc_list.remove(p)

    while num_active_workers < num_max_workers:
        config = my_hpo.get_next_sample()

        if config is None:
            break

        p = Process(target=my_trainer, args=(config, ))
        proc_list.append(p)
        p.start()
        num_active_workers += 1

    # All trials are done.
    if num_active_workers == 0:
        break

print("best hp: ", my_hpo.get_best_config())
