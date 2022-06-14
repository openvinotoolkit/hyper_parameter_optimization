# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import pytest

from hpopt.hpopt import get_cutoff_score, load_json
from hpopt.hyperband import AsyncHyperBand

def test_get_cutoff_score():
    num_imgs = 240
    max_epochs = 300
    min_iters = 200
    batch_size = 4
    max_t = num_imgs * max_epochs
    min_t = batch_size * min_iters
    rf = 2.0
    s = 0

    result_trial8 = load_json('./tests/assets/hpo_results_normal/hpopt_trial_8.json')
    target_ep = 6
    score_at_trial8 = -1
    num_images = -1
    for ep, img in enumerate(result_trial8['images']):
        if ep == target_ep:
            num_images = img
            score_at_trial8 = result_trial8['scores'][ep]
            break
    assert num_images == 31920
    assert score_at_trial8 == 0.0

    rungs = AsyncHyperBand.get_rungs(min_t, max_t, rf, s)
    print(f"rungs = {rungs}")
    save_path = './tests/assets/hpo_results_normal'
    target_rung_idx = -1
    for rung_idx, rung in enumerate(rungs):
        if rung < num_images:
            break
        target_rung_idx = rung_idx
    target_rung = rungs[target_rung_idx]
    print(f"target_rung = {target_rung}")
    mode = 'max'
    cutoff_score = get_cutoff_score(save_path, target_rung, rungs, mode)

    print(f"cutoff_score = {cutoff_score}")
    assert not (cutoff_score > score_at_trial8)
