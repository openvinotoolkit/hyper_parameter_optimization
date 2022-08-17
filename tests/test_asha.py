# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math

import pytest
from hpopt import asha
from hpopt.asha import Trial, Rung, Bracket, HyperBand

@pytest.fixture
def good_trial_args():
    return {"id" : "name", "configuration" : {"hp1" : 1, "hp2" : 1.2}}

@pytest.fixture
def trial(good_trial_args):
    return Trial(**good_trial_args)

@pytest.fixture
def good_rung_args():
    return {"resource" : 10, "num_required_trial" : 16, "reduction_factor" : 2, "rung_idx" : 0}

def register_scores_to_trial(trial, scores = [val for val in range(100)]):
    if len(trial.score) != 0:
        base_resource = max(trial.score.keys())
    else:
        base_resource = 0
    for idx, score in enumerate(scores):
        trial.register_score(score, base_resource+idx+1)

@pytest.fixture
def rung(good_rung_args):
    return Rung(**good_rung_args)

@pytest.fixture
def good_bracket_args():
    hp_configs = [Trial(i, {"hp1" : 1, "hp2" : 1.2}) for i in range(100)]
    return {
        "minimum_resource": 4,
        "maximum_resource": 64,
        "hyper_parameter_configurations" : hp_configs,
        "reduction_factor": 2,
        "mode": "max",
        "asynchronous_sha": True
    }

@pytest.fixture
def bracket(good_bracket_args):
    return Bracket(**good_bracket_args)

@pytest.fixture
def good_hyperband_args():
    return {
        "search_space" : {
            "hp1" : {
                "param_type" : "uniform",
                "max" : 100,
                "min" : 10
            },
            "hp2" : {
                "param_type" : "qloguniform",
                "max" : 1000,
                "min" : 100,
                "step" : 2,
                "log_base" : 10
            }
        },
        "save_path" : "/tmp/hpopt",
        "mode" : "max",
        "num_workers" : 1,
        "num_full_iterations" : 1,
        "non_pure_train_ratio" : 0.2,
        "full_dataset_size" : 100,
        "metric" : "mAP",
        "maximum_resource" : 64,
        "minimum_resource" : 1,
        "reduction_factor" : 2,
        "asynchronous_sha" : True,
        "asynchronous_bracket" : True
    }

@pytest.fixture
def hyper_band(good_hyperband_args):
    return HyperBand(**good_hyperband_args)

@pytest.mark.parametrize("reduction_factor", [4, 100, 4000])
def test_check_reduction_factor_value(reduction_factor):
    asha._check_reduction_factor_value(reduction_factor)

@pytest.mark.parametrize("reduction_factor", [-10, 1])
def test_check_reduction_factor_lesser_value(reduction_factor):
    with pytest.raises(ValueError):
        asha._check_reduction_factor_value(reduction_factor)

class TestTrial:
    def test_init(self, good_trial_args):
        Trial(**good_trial_args)

    @pytest.mark.parametrize("rung_val", [0, 10])
    def teste_set_rung(self, trial, rung_val):
        trial.rung = rung_val

    @pytest.mark.parametrize("rung_val", [-10, -3])
    def test_set_negative_rung(self, trial, rung_val):
        with pytest.raises(ValueError):
            trial.rung = rung_val

    def test_set_iteration(self, trial):
        trial.set_iterations(10)
        assert trial.configuration["iterations"] == 10

    @pytest.mark.parametrize("iter_val", [-10, 0])
    def test_set_negative_iteration(self, trial, iter_val):
        with pytest.raises(ValueError):
            trial.set_iterations(iter_val)

    @pytest.mark.parametrize("score", [-10, 12.5])
    def test_register_score(self, trial, score):
        for resource in [1, 4.3, 10]:
            trial.register_score(score, resource)

    @pytest.mark.parametrize("resource", [-10, 0])
    def test_register_score_not_postive_resource(self, trial, resource):
        score = 10
        with pytest.raises(ValueError):
            trial.register_score(score, resource)

    @pytest.mark.parametrize("mode", ["min", "max"])
    @pytest.mark.parametrize("resource_limit", [None, 10, 20])
    def test_get_best_score(self, trial, mode, resource_limit):
        scores = [val for val in range(100)]
        register_scores_to_trial(trial, scores)

        if resource_limit is not None:
            scores = {i+1 : score for i, score in enumerate(scores)}
            scores = [val for key, val in scores.items() if key <= resource_limit]

        if mode == "min":
            expected_score = min(scores)
        else:
            expected_score = max(scores)

        assert expected_score == trial.get_best_score(mode, resource_limit)

    def test_get_best_score_empty_score(self, trial):
        assert trial.get_best_score() == None

    def test_get_best_score_no_trial_to_meet_condition(self, trial):
        scores = [val for val in range(100)]
        register_scores_to_trial(trial, scores)
        assert trial.get_best_score(resource_limit=0.5) == None

    def test_get_best_score_with_empty_scores(self, trial):
        assert trial.get_best_score() == None
        
    def test_get_best_score_with_wrong_mode_value(self, trial):
        register_scores_to_trial(trial)
        with pytest.raises(ValueError):
            trial.get_best_score(mode="wrong")

    @pytest.mark.parametrize("resource", [12, 42.12])
    def test_get_progress(self, trial, resource):
        trial.register_score(100, resource)
        assert trial.get_progress() == resource

    def test_get_progress_not_trained_at_all(self, trial):
        assert trial.get_progress() == 0

class TestRung:
    def test_init(self, good_rung_args):
        Rung(**good_rung_args)

    @pytest.mark.parametrize("resource", [-10, 0])
    def test_init_resource_nenative(self, good_rung_args, resource):
        wrong_trial_args = good_rung_args
        wrong_trial_args["resource"] = resource
        with pytest.raises(ValueError):
            Rung(**wrong_trial_args)

    @pytest.mark.parametrize("num_required_trial", [-10, 0])
    def test_init_num_required_trial(self, good_rung_args, num_required_trial):
        wrong_trial_args = good_rung_args
        wrong_trial_args["num_required_trial"] = num_required_trial
        with pytest.raises(ValueError):
            Rung(**wrong_trial_args)

    @pytest.mark.parametrize("reduction_factor", [-10, 0, 1])
    def test_init_wrong_reduction_factor(self, good_rung_args, reduction_factor):
        wrong_trial_args = good_rung_args
        wrong_trial_args["reduction_factor"] = reduction_factor
        with pytest.raises(ValueError):
            Rung(**wrong_trial_args)

    @pytest.mark.parametrize("rung_idx", [-10, -3])
    def test_init_wrong_rung_idx(self, good_rung_args, rung_idx):
        wrong_trial_args = good_rung_args
        wrong_trial_args["rung_idx"] = rung_idx
        with pytest.raises(ValueError):
            Rung(**wrong_trial_args)

    def test_add_new_trial(self, rung, good_trial_args):
        for _ in range(rung.num_required_trial):
            trial = Trial(**good_trial_args)
            rung.add_new_trial(trial)
            assert rung._trials[-1] == trial

    def test_add_too_many_trials(self, rung, good_trial_args):
        with pytest.raises(RuntimeError):
            for _ in range(rung.num_required_trial+1):
                trial = Trial(**good_trial_args)
                rung.add_new_trial(trial)

    @pytest.mark.parametrize("mode", ["max", "min"])
    def test_get_best_trial(self, rung, good_trial_args, mode):
        for score in range(rung.num_required_trial):
            trial = Trial(**good_trial_args)
            trial.register_score(score=score, resource=1)
            rung.add_new_trial(trial)

        best_trial = rung.get_best_trial(mode)

        if mode == "max":
            assert  best_trial.get_best_score(mode) == rung.num_required_trial - 1
        else:
            assert  best_trial.get_best_score(mode) == 0

    def test_get_best_trial_with_no_trial(self, rung):
        best_trial = rung.get_best_trial()
        assert best_trial == None

    def test_get_best_trial_wrong_mode_val(self, rung):
        with pytest.raises(ValueError):
            rung.get_best_trial("wrong")

    def test_need_more_trails(self, rung, good_trial_args):
        for i in range(1, rung.num_required_trial+1):
            trial = Trial(**good_trial_args)
            rung.add_new_trial(trial)
            if i != rung.num_required_trial:
                assert rung.need_more_trials() == True
            else:
                assert rung.need_more_trials() == False

    def test_is_done(self, rung, good_trial_args):
        for i in range(rung.num_required_trial-1):
            trial = Trial(**good_trial_args)
            register_scores_to_trial(trial, [val for val in range(rung.resource)])
            rung.add_new_trial(trial)
            assert rung.is_done() == False

        trial = Trial(**good_trial_args)
        register_scores_to_trial(trial, [val for val in range(rung.resource-1)])
        rung.add_new_trial(trial)
        assert rung.is_done() == False
        trial.register_score(100, rung.resource+1)
        assert rung.is_done() == True

    def test_has_promotable_trial_not_asha(self, rung, good_trial_args):
        for i in range(rung.num_required_trial-1):
            trial = Trial(**good_trial_args)
            register_scores_to_trial(trial, [val for val in range(rung.resource)])
            rung.add_new_trial(trial)
        
        assert rung.has_promotable_trial() == False

        trial = Trial(**good_trial_args)
        register_scores_to_trial(trial, [val for val in range(rung.resource)])
        rung.add_new_trial(trial)
        assert rung.has_promotable_trial() == True

        num_promoteable = rung._num_required_trial // rung._reduction_factor
        for _ in range(num_promoteable-1):
            best_trial = rung.get_best_trial()
            best_trial.rung += 1
            assert rung.has_promotable_trial(False) == True

        best_trial = rung.get_best_trial()
        best_trial.rung += 1
        assert rung.has_promotable_trial(False) == False

    def test_has_promotable_trial_asha(self, rung, good_trial_args):
        num_promoteable = rung._num_required_trial // rung._reduction_factor
        for i in range(num_promoteable // rung._reduction_factor):
            for _ in range(rung._reduction_factor):
                trial = Trial(**good_trial_args)
                register_scores_to_trial(trial, [val for val in range(rung.resource)])
                rung.add_new_trial(trial)

            assert rung.has_promotable_trial(True) == True
            best_trial = rung.get_best_trial()
            best_trial.rung += 1
            assert rung.has_promotable_trial(True) == False

    def test_trial_is_done(self, rung, trial):
        register_scores_to_trial(trial, [val for val in range(rung.resource-1)])
        assert rung.trial_is_done(trial) == False
        trial.register_score(100, rung.resource)
        assert rung.trial_is_done(trial) == True

    def test_trial_is_done_trial_not_trained_at_all(self, rung, trial):
        assert rung.trial_is_done(trial) == False

class TestBracket:
    def test_init(self, good_bracket_args):
        Bracket(**good_bracket_args)

    def test_init_minimum_is_negative(self, good_bracket_args):
        wrong_args = good_bracket_args
        wrong_args["minimum_resource"] = -1
        with pytest.raises(ValueError):
            Bracket(**wrong_args)

    @pytest.mark.parametrize("reduction_factor", [-10, 0, 1])
    def test_init_wrong_reduction_factor(self, good_bracket_args, reduction_factor):
        wrong_args = good_bracket_args
        wrong_args["reduction_factor"] = reduction_factor
        with pytest.raises(ValueError):
            Bracket(**wrong_args)

    def test_init_wrong_mode_val(self, good_bracket_args):
        wrong_args = good_bracket_args
        wrong_args["mode"] = "wrong"
        with pytest.raises(ValueError):
            Bracket(**wrong_args)

    def test_init_minimum_val_is_bigger_than_maximum_val(self, good_bracket_args):
        wrong_args = good_bracket_args
        wrong_args["minimum_resource"] = 100
        wrong_args["maximum_resource"] = 10
        with pytest.raises(ValueError):
            Bracket(**wrong_args)

    def test_init_empty_hyper_parameter_configurations(self, good_bracket_args):
        wrong_args = good_bracket_args
        wrong_args["hyper_parameter_configurations"] = []
        with pytest.raises(ValueError):
            Bracket(**wrong_args)

    def test_max_rung(self, good_bracket_args):
        bracket = Bracket(**good_bracket_args)
        expected_val = math.ceil(
            math.log(
                good_bracket_args["maximum_resource"] / good_bracket_args["minimum_resource"],
                good_bracket_args["reduction_factor"]
            )
        )
        assert bracket.max_rung == expected_val

    def test_release_new_trial(self, bracket):
        num_hyper_parameter_configurations = len(bracket.hyper_parameter_configurations)
        new_trial = bracket._release_new_trial()
        assert len(bracket.hyper_parameter_configurations) == num_hyper_parameter_configurations - 1
        assert new_trial.id in bracket._trials

    def test_promote_trial(self, bracket):
        self._make_all_first_rung_trials_done(bracket)

        rung_idx = 0
        while rung_idx < bracket.max_rung:
            if bracket._rungs[rung_idx].has_promotable_trial():
                trial = bracket._promote_trial(rung_idx)
                assert trial != None # If promotable trials exists and not in max_rung, should return trials
                for idx in range(bracket._rungs[rung_idx+1].resource - bracket._rungs[rung_idx].resource):
                    trial.register_score(idx, bracket._rungs[rung_idx].resource + idx + 1)
            elif bracket._rungs[rung_idx].is_done():
                trial = bracket._promote_trial(rung_idx)
                assert trial == None # if promotable trials doesn't exist, return None

                rung_idx += 1
                print(bracket.max_rung, "/", rung_idx)

        trial = bracket._promote_trial(rung_idx)
        assert trial == None # if rung_idx is max_rung, return None

    def _make_all_first_rung_trials_done(self, bracket):
        for _ in range(bracket._rungs[0]._num_required_trial):
            new_trial = bracket._release_new_trial()
            register_scores_to_trial(new_trial, [score for score in range(bracket._rungs[0].resource)])

    def test_prmote_trial_negative_rung_idx(self, bracket):
        with pytest.raises(ValueError):
            bracket._promote_trial(-1)

    def test_register_score(self, bracket):
        new_trial = bracket._release_new_trial()
        registered_score = 100

        bracket.register_score(registered_score, 1, new_trial.id)
        assert new_trial.score[1] == registered_score

    def test_register_score_wrong_id(self, bracket):
        score = 100
        with pytest.raises((ValueError, KeyError)):
            bracket.register_score(score, 1, "wrong_name")

    def test_get_next_trial(self, bracket):
        rung_idx = 0
        while rung_idx <= bracket.max_rung:
            trial_arr = []
            for _ in range(bracket._rungs[rung_idx]._num_required_trial):
                trial = bracket.get_next_trial()
                assert trial != None
                trial_arr.append(trial)

            trial = bracket.get_next_trial()
            assert trial == None

            for trial in trial_arr:
                register_scores_to_trial(
                    trial,
                    [score for score in range(bracket._rungs[rung_idx].resource - trial.get_progress())]
                )

            rung_idx += 1

        trial = bracket.get_next_trial()
        assert trial == None

    def test_is_done(self, bracket):
        while True:
            trial = bracket.get_next_trial()
            if trial is None:
                break

            register_scores_to_trial(
                trial,
                [score for score in range(bracket._rungs[trial.rung].resource - trial.get_progress())]
            )

        assert bracket.is_done() == True

    @pytest.mark.parametrize("num", [1, 5, 15])
    def test_num_trial_is_not_enough(self, good_bracket_args, num):
        wrong_bracket_args = good_bracket_args
        wrong_bracket_args["hyper_parameter_configurations"] = \
            wrong_bracket_args["hyper_parameter_configurations"][:num]

        with pytest.raises(ValueError):
            bracket = Bracket(**wrong_bracket_args)

class TestHyperBand:
    def test_init(self, good_hyperband_args):
        hb = HyperBand(**good_hyperband_args)
        max_bracket_idx = math.floor(
            math.log(
                good_hyperband_args["maximum_resource"] / good_hyperband_args["minimum_resource"],
                good_hyperband_args["reduction_factor"]
            )
        )
        assert hb._max_bracket == max_bracket_idx
        assert len(hb._brackets) == max_bracket_idx + 1

    @pytest.mark.parametrize("minimum_resource", [-10, 0])
    def test_init_not_postive_maximum_resource(self, good_hyperband_args, minimum_resource):
        wrong_arg = good_hyperband_args
        wrong_arg["minimum_resource"] = minimum_resource
        with pytest.raises(ValueError):
            HyperBand(**wrong_arg)

    @pytest.mark.parametrize("reduction_factor", [-10, 0, 1])
    def test_init_wrong_reduction_factor(selfe, good_hyperband_args, reduction_factor):
        wrong_arg = good_hyperband_args
        wrong_arg["reduction_factor"] = reduction_factor
        with pytest.raises(ValueError):
            HyperBand(**wrong_arg)


    @pytest.mark.parametrize("num_brackets", [-10, 0])
    def test_init_not_positive_num_brackets(self, good_hyperband_args, num_brackets):
        wrong_arg = good_hyperband_args
        wrong_arg["num_brackets"] = num_brackets
        with pytest.raises(ValueError):
            HyperBand(**wrong_arg)

    @pytest.mark.parametrize("num", [1, 10])
    def test_make_new_hyper_parameter_configs(self, good_hyperband_args, num):
        hb = HyperBand(**good_hyperband_args)
        trial_arr = hb._make_new_hyper_parameter_configs(num, "test")

        for trial in trial_arr:
            assert "test_" in trial.id
            assert 10 <= trial.configuration["hp1"] <= 100
            assert 100 <= trial.configuration["hp2"] <= 1000
            assert trial.configuration["hp2"] % 2 == 0

    def test_get_next_sample(self, hyper_band):
        while True:
            trial = hyper_band.get_next_sample()
            if trial is None:
                break
            register_scores_to_trial(
                trial,
                [val for val in range(int(trial.configuration["iterations"] - trial.get_progress()))]
            )

        for bracket in hyper_band._brackets:
            assert bracket.is_done() == True

    def test_report_score(self, hyper_band):
        trial = hyper_band.get_next_sample()
        score = 100
        resource = 10
        hyper_band.report_score(score, resource, trial.id)
        assert trial.score[resource] == score

    def test_is_done(self, hyper_band):
        for bracket in hyper_band._brackets:
            assert hyper_band.is_done() == False
            while True:
                trial = bracket.get_next_trial()
                if trial is None:
                    break

                register_scores_to_trial(
                    trial,
                    [score for score in range(int(bracket._rungs[trial.rung].resource) - trial.get_progress())]
                )

        assert hyper_band.is_done() == True