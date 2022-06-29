# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import pytest
# import hpopt

# from torch.utils.data import Dataset
# import os
from hpopt import load_json
from hpopt.hyperband import AsyncHyperBand, _get_cutoff_score, _find_idx_of_max_less_than_target
# def my_model(step, width, height):
#     return (0.1 + width * step / 100)**(-1) + height * 0.1


# def my_trainer(config):
#     width, height = config['params']['width'], config['params']['height']

#     for step in range(config["iterations"]):
#         score = my_model(step, width, height)
        
#         if hpopt.report(config=config, score=score) == hpopt.Status.STOP:
#             break

# def _test_asha():
#     hp_configs = {"width": hpopt.SearchSpace("uniform", [10, 100]),
#                   "height": hpopt.SearchSpace("uniform", [0, 100])}
    
#     my_hpo = hpopt.create(save_path='./tmp/unittest',
#                           search_alg="asha",
#                           search_space=hp_configs,
#                           mode='min',
#                           num_trials=50,
#                           max_iterations=10,
#                           reduction_factor=3,
#                           num_brackets=1,
#                           resume=False,
#                           subset_ratio=1.0,
#                           num_full_iterations=20,
#                           full_dataset_size=1000)

#     assert type(my_hpo) == hpopt.hyperband.AsyncHyperBand

#     config = my_hpo.get_next_sample()

#     assert config['iterations'] > 1
#     assert config['subset_ratio'] > 0.2
#     assert config['file_path'] == './tmp/unittest/hpopt_trial_0.json'

#     my_trainer(config)

#     my_hpo.update_scores()

#     assert my_hpo.hpo_status['config_list'][0]['score'] is not None

#     num_trials_in_brackets = [0, 0, 0, 0, 0]

#     while config is not None:
#         num_trials_in_brackets[config['bracket']] += 1

#         my_trainer(config)

#         config = my_hpo.get_next_sample()

#     assert sum(num_trials_in_brackets) == my_hpo.num_trials

#     best_config = my_hpo.get_best_config()

#     my_hpo.print_results()

#     assert my_model(step=5, **best_config) <= my_model(5, 50, 50)

#     print("best_config: ", best_config)

# if __name__ == '__main__':
#     test_asha()

def test_find_idx_of_max_less_than_target():
    trial_json = {
        "status": 2,
        "scores": [
            0.7077550888061523,
            0.3251700699329376,
            0.5283446907997131,
            0.6163265109062195,
            0.47252747416496277,
            0.60317462682724,
            0.6775510311126709,
            0.75,
            0.7571428418159485,
            0.75,
            0.7687075138092041,
            0.8367347121238708,
            0.7142857313156128,
            0.75,
            0.8571428656578064,
            0.8571428656578064
        ],
        "median": [
            0.7077550888061523,
            0.516462579369545,
            0.5204232831796011,
            0.5443990901112556,
            0.5300247669219971,
            0.5422164102395376,
            0.5615499275071281,
            0.585106186568737,
            0.6042213704850938,
            0.6187992334365845,
            0.6324272589250044,
            0.64945288002491,
            0.6544400224318871,
            0.6612657351153237,
            0.6743242104848226,
            0.6857503764331341
        ],
        "images": [
            4560,
            9120,
            13680,
            18240,
            22800,
            27360,
            31920,
            36480,
            41040,
            45600,
            50160,
            54720,
            59280,
            63840,
            68400,
            72960
        ]
    }
    rungs = [67215, 34447, 18063, 9871, 5775, 3727, 2703, 2191, 1935, 1807, 1743, 1711, 1695, 1687, 1683, 1681, 1680]

    # target is out of range of the rungs
    target_out_of_range = 0
    idx = _find_idx_of_max_less_than_target(trial_json["images"], target_out_of_range)
    print(f"found idx = {idx}")
    assert idx == -1

    target_in_range = rungs[4]
    idx = _find_idx_of_max_less_than_target(trial_json["images"], target_in_range)
    print(f"found idx = {idx}")
    assert idx != -1

    target_exact_matched = 31920
    idx = _find_idx_of_max_less_than_target(trial_json["images"], target_exact_matched)
    print(f"found idx = {idx}")
    assert idx == trial_json["images"].index(target_exact_matched)


def test_get_cutoff_score_with_result_json():
    num_imgs = 240
    max_epochs = 300
    min_epochs = 7
    rf = 2.0
    s = 0

    save_path = './tests/assets/hpo_results_1'
    config_list = load_json(os.path.join(save_path, 'hpopt_status.json'))['config_list']

    trial = 1
    trial_result = load_json(os.path.join(save_path, f'hpopt_trial_{trial}.json'))
    max_t = num_imgs * max_epochs
    min_t = num_imgs * min_epochs
    target_ep = 0
    target_score = -1
    num_images = -1
    for ep, img in enumerate(trial_result['images']):
        if ep == target_ep:
            num_images = img
            target_score = trial_result['scores'][ep]
            break
    assert num_images == 4560
    assert target_score == 0.005
    rungs = AsyncHyperBand.get_rungs(min_t, max_t, rf, s)
    print(f"rungs = {rungs}")
    target_rung_idx = -1
    for rung_idx, rung in enumerate(rungs):
        if rung < num_images:
            break
        target_rung_idx = rung_idx
    target_rung = rungs[target_rung_idx]
    print(f"target_rung = {target_rung}")
    mode = 'max'
    cutoff_score = _get_cutoff_score(save_path, target_rung, rungs, mode)

    print(f"cutoff_score for rung {target_rung} = {cutoff_score}")
    assert not (cutoff_score > target_score), f"cutoff score = {cutoff_score}, score @ trial{trial}[{target_ep}] = {target_score}"

    trial = 6
    trial_result = load_json(os.path.join(save_path, f'hpopt_trial_{trial}.json'))
    # trial_config = get_trial_config(config_list, trial)
    # assert trial_config is not None
    # batch_size = trial_config['config']['learning_parameters.batch_size']
    # print(f"batch_size for trial{trial} = {batch_size}")
    # max_t = num_imgs * max_epochs
    # min_t = num_imgs * min_epochs
    target_ep = 10
    target_score = -1
    num_images = -1
    for ep, img in enumerate(trial_result['images']):
        if ep == target_ep:
            num_images = img
            target_score = trial_result['scores'][ep]
            break
    assert num_images == 36504, f"num images = {num_images}"
    assert target_score == 0.23577335476875305, f"score @ trial{trial} = {target_score}"
    # rungs = AsyncHyperBand.get_rungs(min_t, max_t, rf, s)
    print(f"rungs = {rungs}")
    target_rung_idx = -1
    for rung_idx, rung in enumerate(rungs):
        target_rung_idx = rung_idx
        if rung < num_images:
            break
    target_rung = rungs[target_rung_idx]
    print(f"target_rung[{target_rung_idx}] = {target_rung}")
    mode = 'max'
    cutoff_score = get_cutoff_score(save_path, target_rung, rungs, mode)
    print(f"cutoff_score for rung {target_rung} = {cutoff_score}")
    assert cutoff_score > target_score, f"cutoff score = {cutoff_score}, score @ trial{trial}[{target_ep}] = {target_score}"

