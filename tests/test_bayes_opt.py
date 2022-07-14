# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import unittest
import hpopt
from hpopt.utils import HpoDataset

from torch.utils.data import Dataset
import os


class my_dataset(Dataset):
    def __getitem__(self, index):
        return [5678]

    def __len__(self):
        return 2000


def my_model(a, b, c, d, e):
    return a * b + c * d - e


def my_trainer(config):
    full_dataset = my_dataset()
    new_dataset = HpoDataset(full_dataset, config)

    # assert len(new_dataset) == 1800
    assert new_dataset[234][0] == 5678

    for iteration_num in range(config["iterations"]):
        score = my_model(**config["params"])
        score += iteration_num
        retval = hpopt.report(config=config, score=score)

        assert (retval == hpopt.Status.STOP) or (retval == hpopt.Status.RUNNING)

        if retval == hpopt.Status.STOP:
            break

class TestBayesOpt(unittest.TestCase):
    def setUp(self):
        self.output = './tmp/unittests'
        self.hp_configs = {
            'a': hpopt.SearchSpace("uniform", [-5, 10]),
            'b': hpopt.SearchSpace("quniform", [2, 14, 2]),
            'c': hpopt.SearchSpace("loguniform", [0.0001, 0.1]),
            'd': hpopt.SearchSpace("qloguniform", [2, 256, 2]),
            'e': hpopt.SearchSpace("choice", [98, 765, 4, 321])
        }

    def tearDown(self):
        try:
            os.remove(hpopt.get_status_path(self.output))
        except FileNotFoundError:
            pass
        # pass

    def test_bayes_opt_resume(self):
        init_trials = remained_trials = 5
        my_hpo = hpopt.create(save_path=self.output,
                            search_alg="bayes_opt",
                            search_space=self.hp_configs,
                            early_stop="median_stop",
                            num_init_trials=init_trials,
                            num_trials=7,
                            max_iterations=2,
                            subset_ratio=0.9,
                            image_resize=[123, 456],
                            resume=False,
                            num_full_iterations=1,
                            full_dataset_size=1)

        assert type(my_hpo) == hpopt.smbo.BayesOpt

        config = my_hpo.get_next_sample()

        assert config['iterations'] == 2
        assert config['file_path'] == os.path.join(self.output, 'hpopt_trial_0.json')
        assert config['early_stop'] is None
        assert config['subset_ratio'] == 0.9
        assert config['resize_width'] == 123
        assert config['resize_height'] == 456

        my_trainer(config)
        remained_trials -= 1
        my_hpo.update_scores()
        assert my_hpo.hpo_status['config_list'][0]['score'] is not None, f"hpo_status = {my_hpo.hpo_status}"
        del my_hpo

        # create same thing and check whether it resume correctly
        my_hpo = hpopt.create(save_path=self.output,
                            search_alg="bayes_opt",
                            search_space=self.hp_configs,
                            early_stop="median_stop",
                            num_init_trials=init_trials,
                            num_trials=7,
                            max_iterations=2,
                            subset_ratio=0.9,
                            image_resize=[123, 456],
                            resume=True,
                            num_full_iterations=1,
                            full_dataset_size=1)

        assert type(my_hpo) == hpopt.smbo.BayesOpt

        configs = my_hpo.get_next_samples()

        assert len(configs) == remained_trials

        # for config in configs:
        #     my_trainer(config)
        #     init_trials -= 1

        # configs = my_hpo.get_next_samples()

        # assert len(configs) == 1

        # my_trainer(configs[0])

        # config = my_hpo.get_next_sample()

        # assert config is not None

        # my_trainer(config)

        # config = my_hpo.get_next_sample()

        # assert config is None

        
        # del my_hpo
        # my_hpo = hpopt.create(save_path=self.output,
        #                     search_alg="bayes_opt",
        #                     search_space=hp_configs,
        #                     ealry_stop="median_stop",
        #                     num_init_trials=5,
        #                     num_trials=2,
        #                     max_iterations=2,
        #                     subset_ratio=0.9,
        #                     image_resize=(123, 456),
        #                     resume=False,
        #                     num_full_iterations=1,
        #                     full_dataset_size=1)

        # assert os.path.exists(hpopt.get_status_path(self.output))

    def _test_bayes_opt(self):
        my_hpo = hpopt.create(save_path=self.output,
                            search_alg="bayes_opt",
                            search_space=self.hp_configs,
                            #ealry_stop="median_stop",
                            #num_init_trials=5,
                            #num_trials=2,
                            #max_iterations=2,
                            #subset_ratio=0.9,
                            #image_resize=(123, 456),
                            resume=False,
                            num_full_iterations=20,
                            full_dataset_size=1000)

        assert my_hpo is not None

        while True:
            config = my_hpo.get_next_sample()

            if config is None:
                break

            my_trainer(config)

        best_config = my_hpo.get_best_config()

        my_hpo.print_results()

        assert my_model(**best_config) >= my_model(3, 8, 0.01, 128, 98)

        print("best_config: ", best_config)