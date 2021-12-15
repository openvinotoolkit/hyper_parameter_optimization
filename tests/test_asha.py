import pytest
import hpopt

from torch.utils.data import Dataset
import os


def my_model(step, width, height):
    return (0.1 + width * step / 100)**(-1) + height * 0.1


def my_trainer(config):
    width, height = config['params']['width'], config['params']['height']

    for step in range(config["iterations"]):
        score = my_model(step, width, height)
        
        if hpopt.report(config=config, score=score) == hpopt.Status.STOP:
            break

def test_asha():
    hp_configs = {"width": hpopt.search_space("uniform", [10, 100]),
                  "height": hpopt.search_space("uniform", [0, 100])}
    
    my_hpo = hpopt.create(save_path='./tmp/unittest',
                          search_alg="asha",
                          search_space=hp_configs,
                          mode='min',
                          num_trials=50,
                          max_iterations=10,
                          reduction_factor=3,
                          num_brackets=1,
                          resume=False,
                          subset_ratio=1.0,
                          num_full_iterations=20,
                          full_dataset_size=1000)

    assert type(my_hpo) == hpopt.asha.AsyncHyperBand

    config = my_hpo.get_next_sample()

    assert config['iterations'] > 1
    assert config['subset_ratio'] > 0.2
    assert config['file_path'] == './tmp/unittest/hpopt_trial_0.json'

    my_trainer(config)

    my_hpo.update_scores()

    assert my_hpo.hpo_status['config_list'][0]['score'] is not None

    num_trials_in_brackets = [0, 0, 0, 0, 0]

    while config is not None:
        num_trials_in_brackets[config['bracket']] += 1

        my_trainer(config)

        config = my_hpo.get_next_sample()

    assert sum(num_trials_in_brackets) == my_hpo.num_trials

    best_config = my_hpo.get_best_config()

    my_hpo.print_results()

    assert my_model(step=5, **best_config) <= my_model(5, 50, 50)

    print("best_config: ", best_config)

if __name__ == '__main__':
    test_asha()
