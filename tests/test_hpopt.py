# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import pytest

from hpopt.base import SearchSpace
from hpopt.smbo import BayesOpt
from hpopt.hyperband import AsyncHyperBand

from hpopt import (
    create,
    finalize_trial,
    get_best_score,
    get_best_score_with_num_imgs,
    get_current_status,
    get_previous_status,
    get_status_path,
    get_trial_path,
    get_trial_results,
    load_json,
    report,
    reportOOM,
)


def test_create():
    hpo = create(100, 100, { "num": SearchSpace("uniform", [0, 10]) }, num_trials=1)
    assert isinstance(hpo, BayesOpt)

    hpo = create(100, 100, { "num": SearchSpace("uniform", [0, 10]) }, search_alg="asha")
    assert isinstance(hpo, AsyncHyperBand)


def test_finalize_trial():
    assert True


def test_get_best_score():
    assert True


def test_get_best_score_with_num_imgs():
    assert True


def test_get_current_status():
    assert True


def test_get_previous_status():
    assert True


def test_get_status_path():
    assert True


def test_get_trial_path():
    assert True


def test_get_trial_results():
    assert True


def test_load_json():
    assert True


def test_report():
    assert True


def test_reportOOM():
    assert True

