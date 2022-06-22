# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import pytest

from hpopt.hpopt import get_cutoff_score, load_json
from hpopt.hyperband import AsyncHyperBand

def get_trial_config(config_list, trial):
    for config in config_list:
        if config['trial_id'] == trial:
            return config
    return None

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
    cutoff_score = get_cutoff_score(save_path, target_rung, rungs, mode)

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
    target_ep = 8
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
        if rung < num_images:
            break
        target_rung_idx = rung_idx
    target_rung = rungs[target_rung_idx]
    print(f"target_rung[{target_rung_idx}] = {target_rung}")
    mode = 'max'
    cutoff_score = get_cutoff_score(save_path, target_rung, rungs, mode)
    print(f"cutoff_score for rung {target_rung} = {cutoff_score}")
    assert cutoff_score > target_score, f"cutoff score = {cutoff_score}, score @ trial{trial}[{target_ep}] = {target_score}"