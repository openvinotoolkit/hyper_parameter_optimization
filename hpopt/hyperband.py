import math
import os
import json
from os import path as osp
from typing import Any, Dict, List, Optional, Union

from pyDOE.doe_lhs import lhs as latin_hypercube_sample

from hpopt.hpo_base import HpoBase, Trial
from hpopt.logger import get_logger
from hpopt.utils import (
    check_mode_input,
    check_positive,
    check_not_negative,
    left_is_better,
)

logger = get_logger()


def _check_reduction_factor_value(reduction_factor: int):
    if reduction_factor < 2:
        raise ValueError(
            "reduction_factor should be at least 2.\n"
            f"your value : {reduction_factor}"
            )

class AshaTrial(Trial):
    def __init__(
        self,
        id: Any,
        configuration: Dict,
        train_environment: Optional[Dict] = None
    ):
        super().__init__(id, configuration, train_environment)
        self._rung = 0

    @property
    def rung(self):
        return self._rung

    @rung.setter
    def rung(self, val: int):
        check_not_negative(val, "rung")
        self._rung = val

    def save_results(self, save_path: str):
        results = {
            "id" : self.id,
            "rung" : self.rung,
            "configuration" : self.configuration,
            "train_environment" : self.train_environment,
            "score" : {resource : score for resource, score in self.score.items()}
        }

        with open(save_path, "w") as f:
            json.dump(results, f)


class Rung:
    def __init__(
        self,
        resource: int,
        num_required_trial: int,
        reduction_factor: int,
        rung_idx: int,
    ):
        check_positive(resource, "resource")
        check_positive(num_required_trial, "num_required_trial")
        _check_reduction_factor_value(reduction_factor)
        check_not_negative(rung_idx, "rung_idx")

        self._reduction_factor = reduction_factor
        self._num_required_trial = num_required_trial
        self._resource = resource
        self._trials: List[AshaTrial] = []
        self._rung_idx = rung_idx

    @property
    def num_required_trial(self):
        return self._num_required_trial

    @property
    def resource(self):
        return self._resource

    @property
    def rung_idx(self):
        return self._rung_idx

    def add_new_trial(self, trial: AshaTrial):
        if not self.need_more_trials():
            raise RuntimeError(f"{self.rung_idx} rung has already sufficient trials.")
        trial.iteration = self.resource
        trial.rung = self.rung_idx
        self._trials.append(trial)

    def get_best_trial(self, mode: str = "max"):
        check_mode_input(mode)
        best_score = None
        best_trial = None
        for trial in self._trials:
            if trial.rung != self.rung_idx:
                continue
            trial_score = trial.get_best_score(mode, self.resource)
            if (
                trial_score is not None
                and (best_score is None or left_is_better(trial_score, best_score, mode))
            ):
                best_trial = trial
                best_score = trial_score

        return best_trial

    def need_more_trials(self):
        return self.num_required_trial > self.get_num_trials_started()

    def get_num_trials_started(self):
        return len(self._trials)

    def is_done(self):
        if self.need_more_trials():
            return False
        for trial in self._trials:
            if not self.trial_is_done(trial):
                return False
        return True

    def get_trial_to_promote(self, asynchronous_sha: bool = False, mode: str = "max"):
        num_finished_trial = 0
        num_promoted_trial = 0
        best_score = None
        best_trial = None

        for trial in self._trials:
            if trial.rung == self._rung_idx:
                if self.trial_is_done(trial):
                    num_finished_trial += 1
                    trial_score = trial.get_best_score(mode, self.resource)
                    if best_score is None or left_is_better(trial_score, best_score, mode):
                        best_trial = trial
                        best_score = trial_score
            else:
                num_promoted_trial += 1

        if asynchronous_sha:
            if (num_promoted_trial + num_finished_trial) // self._reduction_factor > num_promoted_trial:
                return best_trial
        else:
            if (
                self.is_done()
                and self._num_required_trial // self._reduction_factor > num_promoted_trial
            ):
                return best_trial

        return None

    def trial_is_done(self, trial: AshaTrial):
        return trial.get_progress() >= self.resource

class Bracket:
    def __init__(
        self,
        minimum_resource: Union[float, int],
        maximum_resource: Union[float, int],
        hyper_parameter_configurations: List[AshaTrial],
        reduction_factor: int = 3,
        mode: str = "max",
        asynchronous_sha: bool = True
    ):
        check_positive(minimum_resource, "minimum_resource")
        _check_reduction_factor_value(reduction_factor)
        check_mode_input(mode)

        self._minimum_resource = minimum_resource
        self.maximum_resource = maximum_resource
        self.hyper_parameter_configurations = hyper_parameter_configurations
        self._reduction_factor = reduction_factor
        self._mode = mode
        self._asynchronous_sha = asynchronous_sha
        self._num_trials = len(hyper_parameter_configurations)

        minimum_num_trials = self._reduction_factor ** self.max_rung
        if minimum_num_trials > self._num_trials:
            raise ValueError(
                "number of hyper_parameter_configurations is not enough. "
                f"minimum number is {minimum_num_trials}, but current number is {self._num_trials}. "
                "if you want to let them be, you can decrease needed number "
                "by increasing reduction factor or minimum resource."
            )

        self._rungs: List[Rung] = [
            Rung(
                minimum_resource * (self._reduction_factor ** idx),
                math.floor(self._num_trials * (self._reduction_factor ** -idx)),
                self._reduction_factor,
                idx,
            ) for idx in range(self.max_rung + 1)
        ]
        self._trials: Dict[int, AshaTrial] = {}

    @property
    def maximum_resource(self):
        return self._maximum_resource

    @maximum_resource.setter
    def maximum_resource(self, val: Union[float, int]):
        check_positive(val, "maximum_resource")
        if val < self._minimum_resource:
            raise ValueError(
                "maxnum_resource should be greater than minimum_resource.\n"
                f"value to set : {val}, minimum_resource : {self._minimum_resource}"
            )
        elif val == self._minimum_resource: 
            logger.warning("maximum_resource is same with the minimum_resource.")

        self._maximum_resource = val

    @property
    def hyper_parameter_configurations(self):
        return self._hyper_parameter_configurations
    
    @hyper_parameter_configurations.setter
    def hyper_parameter_configurations(self, val: List[AshaTrial]):
        if len(val) == 0:
            raise ValueError("hyper_parameter_configurations should have at least one element.")
        self._hyper_parameter_configurations = val

    @property
    def max_rung(self):
        return math.ceil(
            math.log(
                self.maximum_resource / self._minimum_resource,
                self._reduction_factor
            )
        )

    def _release_new_trial(self):
        if not self.hyper_parameter_configurations:
            return None

        new_trial = self.hyper_parameter_configurations.pop(0)
        self._rungs[0].add_new_trial(new_trial)
        self._trials[new_trial.id] = new_trial

        return new_trial

    def _promote_trial_if_available(self, rung_idx: int):
        check_not_negative(rung_idx, "rung_idx")

        if self.max_rung <= rung_idx:
            return None

        best_trial = self._rungs[rung_idx].get_trial_to_promote(self._asynchronous_sha, self._mode)
        if best_trial is not None:
            self._rungs[rung_idx+1].add_new_trial(best_trial)

        return best_trial

    def register_score(self, score: Union[float, int], resource: Union[float, int], trial_id: Any):
        self._trials[trial_id].register_score(score, resource)

    def get_next_trial(self):
        next_sample = None
        for current_rung in range(self.max_rung-1, -1, -1):
            next_sample = self._promote_trial_if_available(current_rung)
            if next_sample is not None:
                break

        if next_sample is None:
            next_sample = self._release_new_trial()

        return next_sample

    def is_done(self):
        return self._rungs[-1].is_done()

    def get_best_trial(self):
        if not self.is_done():
            logger.warning("Bracket is not done yet.")

        trial = None
        for rung in reversed(self._rungs):
            trial = rung.get_best_trial(self._mode)
            if trial is None:
                continue
            break

        return trial

    def save_results(self, save_path: str):
        result = {
            "minimum_resource" : self._minimum_resource,
            "maximum_resource" : self.maximum_resource,
            "reduction_factor" : self._reduction_factor,
            "mode" : self._mode,
            "asynchronous_sha" : self._asynchronous_sha,
            "num_trials" : self._num_trials,
            "rung_status" : [
                {
                    "rung_idx" : rung.rung_idx,
                    "num_trial_started" : rung.get_num_trials_started(),
                    "num_required_trial" : rung.num_required_trial,
                    "resource" : rung.resource

                } for rung in self._rungs
            ]
        }
        with open(osp.join(save_path, "rung_status.json"), "w") as f:
            json.dump(result, f)

        for trial_id, trial in self._trials.items():
            trial.save_results(osp.join(save_path, f"{trial_id}.json"))

    def report_trial_is_done(self, trial_id: Any):
        trial = self._trials[trial_id]
        if trial.get_progress() < trial.iteration:
            best_score = trial.get_best_score()
            if best_score is None:
                raise RuntimeError(f"Although {trial_id} trial doesn't report any score but it's done")
            self.register_score(best_score, trial.iteration, trial_id)

class HyperBand(HpoBase):
    """
    This implements the Asyncronous HyperBand scheduler with iterations only.
    Please refer the below papers for the detailed algorithm.

    [1] "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization", JMLR 2018
        https://arxiv.org/abs/1603.06560
        https://homes.cs.washington.edu/~jamieson/hyperband.html

    [2] "A System for Massively Parallel Hyperparameter Tuning", MLSys 2020
        https://arxiv.org/abs/1810.05934

    Args:
        min_iterations (int): Only stop trials at least this old in time.
            The units are the same as the attribute named by `time_attr`.
        reduction_factor (float): Used to set halving rate and amount. This
            is simply a unit-less scalar.
        num_brackets (int): Number of brackets. Each bracket has a different
            halving rate, specified by the reduction factor.
    """

    def __init__(
        self,
        minimum_resource: Optional[Union[int, float]] = None,
        reduction_factor: int = 3,
        num_brackets: Optional[int] = None,
        asynchronous_sha: bool = True,
        asynchronous_bracket: bool = False,
        **kwargs
    ):
        super(HyperBand, self).__init__(**kwargs)

        if minimum_resource is None:
            pass
        else:
            check_positive(minimum_resource, "minimum_resource")
        _check_reduction_factor_value(reduction_factor)

        self._next_trial_id = 0
        self._reduction_factor = reduction_factor
        self._minimum_resource = minimum_resource
        self._asynchronous_sha = asynchronous_sha
        self._asynchronous_bracket = asynchronous_bracket
        if num_brackets is not None:
            check_positive(num_brackets, "num_brackets")
            self._num_bracket = num_brackets
        else:
            self._num_bracket = self._calculate_num_bracket()
        self._brackets: List[Bracket] = self._make_brackets()

    def _calculate_num_bracket(self):
        return math.floor(math.log(self.maximum_resource / self._minimum_resource, self._reduction_factor)) + 1

    def _make_brackets(self):
        """
        bracket order is the opposite of order of paper's.
        this is for running default hyper parmeters with abundant resource.
        """
        brackets = []

        for idx in range(self._num_bracket):
            num_bracket_trials = math.ceil(
                self._num_bracket
                * (self._reduction_factor ** idx)
                / (idx + 1)
            )
            configurations = self._make_new_hyper_parameter_configs(num_bracket_trials, str(idx))
            bracket = Bracket(
                self.maximum_resource * (self._reduction_factor ** -idx),
                self.maximum_resource,
                configurations,
                self._reduction_factor,
                self.mode,
                self._asynchronous_sha
            )
            brackets.append(bracket)

        return brackets

    def _make_new_hyper_parameter_configs(
        self,
        num: int,
        trial_id_prefix: Optional[str] = None
    ):
        check_positive(num, "num")

        if trial_id_prefix is not None:
            trial_id_prefix = trial_id_prefix + "_"
        else:
            trial_id_prefix = ""
        hp_configs = []

        if self.prior_hyper_parameters is not None:
            hp_configs.extend(self._get_prior_hyper_parameters(num, trial_id_prefix))

        hp_configs.extend(self._get_random_hyper_parameter(num-len(hp_configs), trial_id_prefix))

        return hp_configs

    def _get_prior_hyper_parameters(self, num_samples: int, trial_id_prefix: str):
        hp_configs = []
        num_samples = min([num_samples, len(self.prior_hyper_parameters)])
        for _ in range(num_samples):
            hyper_parameter = self.prior_hyper_parameters.pop(0)
            hp_configs.append(self._make_trial(trial_id_prefix + self._get_new_trial_id(), hyper_parameter))

        return hp_configs

    def _get_random_hyper_parameter(self, num_samples: int, trial_id_prefix: str):
        hp_configs = []
        configurations = latin_hypercube_sample(len(self.search_space), num_samples)
        for config in configurations:
            config_with_key = {key : config[idx] for idx, key in enumerate(self.search_space)}
            hp_configs.append(
                self._make_trial(
                    trial_id_prefix + self._get_new_trial_id(),
                    self.search_space.convert_from_zero_one_scale_to_real_space(config_with_key),
                )
            )

        return hp_configs

    def _make_trial(self, id: str, hyper_parameter: Dict):
        return AshaTrial(id, hyper_parameter, self._get_train_environment())

    def _get_new_trial_id(self):
        id = self._next_trial_id
        self._next_trial_id += 1
        return str(id)

    def _get_train_environment(self):
        train_environment = {"subset_ratio" : self.subset_ratio}
        return train_environment

    def get_next_sample(self):
        next_sample = None
        for bracket in self._brackets:
            if not bracket.is_done():
                next_sample = bracket.get_next_trial()
                if self._asynchronous_bracket and next_sample is None:
                    continue
                break

        return next_sample

    def save_results(self):
        for idx, bracket in enumerate(self._brackets):
            save_path = osp.join(self.save_path, str(idx))
            os.makedirs(save_path, exist_ok=True)
            bracket.save_results(save_path)

    def auto_config(self):
        raise NotImplementedError

    def get_progress(self):
        raise NotImplementedError

    def report_score(self, score: Union[float, int], resource: Union[float, int], trial_id: str, done: bool = False):
        bracket_idx = trial_id.split('_')[0]
        if done:
            self._brackets[int(bracket_idx)].report_trial_is_done(trial_id)
        else:
            self._brackets[int(bracket_idx)].register_score(score, resource, trial_id)

    def is_done(self):
        for bracket in self._brackets:
            if not bracket.is_done():
                return False
        return True

    def get_best_config(self):
        best_score = None
        best_trial = None

        for bracket in self._brackets:
            trial = bracket.get_best_trial()
            if trial is None:
                continue

            score = trial.get_best_score()
            if best_score is None or left_is_better(score, best_score, self.mode):
                best_score = score
                best_trial = trial

        if best_trial is None:
            return None
        return best_trial.configuration
