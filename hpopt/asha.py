import math
from typing import Any, Dict, List, Optional, Union

from pyDOE.doe_lhs import lhs as latin_hypercube_sample

from hpopt.hpo_base import HpoBase
from hpopt.logger import get_logger
from hpopt.utils import (
    _check_mode_input,
    _check_positive,
    _check_type,
    _left_is_better,
)

logger = get_logger()


class Trial:
    def __init__(
        self,
        id: Any,
        configuration: Dict
    ):
        self._id = id
        self._configuration = configuration
        self._rung = 0
        self.score = []

    @property
    def id(self):
        return self._id

    @property
    def configuration(self):
        return self._configuration

    @property
    def rung(self):
        return self._rung

    @rung.setter
    def rung(self, val: int):
        _check_type(val, int, "rung")
        _check_positive(val, "rung")
        self._rung = val
    
    def set_iterations(self, iter: int):
        _check_type(iter, int, "iter")
        _check_positive(iter, "iter")
        self._configuration["iterations"] = iter

    def append_score(self, val: Union[float, int]):
        _check_type(val, (float, int), "score")
        self.score.append(val)

    def get_best_score(self, mode: str ="max", idx_limit: Optional[int] = None):
        if idx_limit is not None:
            score_arr = self.score[:idx_limit]
        _check_mode_input(mode)
        if mode == "max":
            return max(score_arr)
        else:
            return min(score_arr)

    def rise_in_rung(self, new_iter: int):
        self.rung = self.rung + 1
        self.set_iterations(new_iter)

def _check_reduction_factor_value(reduction_factor: int):
    if reduction_factor < 2:
        raise ValueError(
            "reduction_factor should be at least 2.\n"
            f"your value : {reduction_factor}"
            )

class Rung:
    def __init__(
        self,
        resource: Union[int, float],
        num_required_trial: int,
        reduction_factor: int,
        rung_idx: int,
        asynchronous_sha: bool = False
    ):
        _check_type(resource, (int, float))
        _check_positive(resource, "resource")
        _check_type(num_required_trial, int, "num_required_trial")
        _check_positive(num_required_trial, "num_required_trial")
        _check_reduction_factor_value(reduction_factor)
        _check_type(rung_idx, int, "rung_idx")
        _check_positive(rung_idx, "rung_idx")
        _check_type(asynchronous_sha, bool, "asynchronous_sha")

        self._reduction_factor = reduction_factor
        self._asynchronous_sha = asynchronous_sha
        self._num_required_trial = num_required_trial
        self._resource = resource
        self._trials: List[Trial] = []
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

    def add_new_trial(self, trial: Trial):
        _check_type(trial, Trial, "trial")
        self._trials.append(trial)

    def get_best_trial(self, mode: str ="max"):
        _check_mode_input(mode)
        best_score = None
        best_trial = None
        for trial in self._trials:
            if trial.rung != self.rung_idx:
                continue
            trial_score = trial.get_best_score(mode, self.resource)
            if best_score is None or _left_is_better(trial_score, best_score, mode):
                best_trial = trial
                best_score = trial_score

        return best_trial

    def need_more_trials(self):
        return self.num_required_trial != len(self._trials)

    def is_done(self):
        if self.need_more_trials():
            return False
        for trial in self._trials:
            if len(trial.score) < self._resource:
                return False
        return True

    def has_promotable_trial(self):
        num_finished_trial = 0
        num_promoted_trial = 0
        for trial in self._trials:
            if trial.rung == self._rung_idx:
                if len(trial.score) >= self._resource:
                    num_finished_trial += 1
            else:
                num_promoted_trial += 1

        if self._asynchronous_sha:
            return num_finished_trial // self._reduction_factor > num_promoted_trial
        else:
            return (
                self._is_done()
                and self._num_required_trial // self._reduction_factor > num_promoted_trial
            )

class Bracket:
    def __init__(
        self,
        minimum_resource: Union[float, int],
        maximum_resource: Union[float, int],
        hyper_parameter_configurations: List[Trial],
        reduction_factor: int = 3,
        mode: str = "max",
        asynchronous_sha: bool = True
    ):
        _check_type(minimum_resource, (float, int), "minimum_resource")
        _check_positive(minimum_resource, "minimum_resource")
        _check_reduction_factor_value(reduction_factor)
        _check_mode_input(mode)
        _check_type(asynchronous_sha, bool, "asynchronous_sha")

        self._minimum_resource = minimum_resource
        self.maximum_resource = maximum_resource
        self.hyper_parameter_configurations = hyper_parameter_configurations
        self._reduction_factor = reduction_factor
        self._mode = mode
        self._asynchronous_sha = asynchronous_sha
        self._num_trials = len(hyper_parameter_configurations)
        self._rungs: List[Rung] = [
            Rung(
                minimum_resource * (self._reduction_factor ** idx),
                math.floor(self._num_trials * (self._reduction_factor ** -idx)),
                self._reduction_factor,
                idx,
                self._asynchronous_sha
            ) for idx in range(self.max_rung + 1)
        ]
        self._trials: Dict[int, Trial] = {}

    @property
    def maximum_resource(self):
        return self._maximum_resource

    @maximum_resource.setter
    def maximum_resource(self, val: Union[float, int]):
        _check_type(val, (float, int), "maximum_resource")
        _check_positive(val, "maximum_resource")
        if val > self._minimum_resource:
            raise ValueError(
                "maxnum_resource is greater than minimum_resource.\n"
                f"value to set : {val}, minimum_resource : {self._minimum_resource}"
            )
        elif val == self._minimum_resource: 
            logger.warning("maximum_resource is same with the minimum_resource.")

        self._maximum_resource = val

    @property
    def hyper_parameter_configurations(self):
        return self._hyper_parameter_configurations
    
    @hyper_parameter_configurations.setter
    def hyper_parameter_configurations(self, val: List[Trial]):
        if len(val) == 0:
            raise ValueError(
                "hyper_parameter_configurations should have at least one element."
            )
        self._hyper_parameter_configurations = val

    @property
    def max_rung(self):
        return math.ceil(
            math.log(
                self.maximum_resource / self._minimum_resource,
                self._reduction_factor
            )
        )

    def _run_new_trial(self):
        if not self.hyper_parameter_configurations:
            return None

        new_trial = self.hyper_parameter_configurations.pop(0)
        new_trial.set_iterations(self._rungs[0].resource)

        self._rungs[0].add_new_trial(new_trial)
        self._trials[new_trial.id] = new_trial

        return new_trial

    def _promote_trial(self, rung_idx: int):
        if (not self._rungs[rung_idx].has_promotable_trial()
            or self.max_rung == rung_idx
        ):
            return None
        best_trial = self._rungs[rung_idx].get_best_trial(self._mode)
        best_trial.rise_in_rung(self._rungs[rung_idx+1].resource)
        return best_trial

    def report_score(self, score: Union[float, int], trial_id: Any):
        _check_type(score, (float, int),  "score")
        self._trials[trial_id].append_score(score)

    def get_next_sample(self):
        next_sample = None
        for current_rung in range(self.max_rung-1, -1, -1):
            if self._rungs[current_rung].has_promotable_trial():
                next_sample = self._promote_trial(current_rung)
                break
        if next_sample is None:
            next_sample = self._run_new_trial()

        return next_sample

    def is_done(self):
        return self._rungs[-1].is_done()

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
        minimum_resource: Union[int, float] = 1,
        reduction_factor: int = 3,
        num_brackets: Optional[int] = None,
        asynchronous_sha: bool = True,
        asynchronous_bracket: bool = False,
        **kwargs,
    ):
        super(HyperBand, self).__init__(**kwargs)

        _check_type(minimum_resource, (float, int), "minimum_resource")
        _check_positive(minimum_resource, "minimum_resource")
        _check_reduction_factor_value(reduction_factor)
        _check_type(asynchronous_sha, bool, "asynchronous_sha")
        _check_type(asynchronous_bracket, bool, "asynchronous_sha")

        if num_brackets is not None:
            _check_type(num_brackets, int, "num_brackets")
            _check_positive(num_brackets, "num_brackets")
            self._max_bracket = num_brackets - 1
        else:
            self._max_bracket = math.ceil(
                math.log(
                    self.maximum_resource / self._minimum_resource,
                    self._reduction_factor
                )
            )
        self._reduction_factor = reduction_factor
        self._minimum_resource = minimum_resource
        self._asynchronous_sha = asynchronous_sha
        self._asynchronous_bracket = asynchronous_bracket
        self._brackets: List[Bracket] = []
        # bracket order is the opposite of order of paper's.
        # this is for running default hyper parmeters with abundnat resource.
        for idx in range(self._max_bracket + 1):
            num_bracket_trials = math.ceil(
                (self._max_bracket + 1)
                * (self._reduction_factor ** idx)
                / (idx + 1)
            )
            configurations = self._get_new_hyper_parameter_configs(num_bracket_trials, idx)
            bracket = Bracket(
                self.maximum_resource * (self._reduction_factor ** -idx),
                self.maximum_resource,
                configurations,
                self._reduction_factor,
                self.mode,
                self._asynchronous_sha
            )
            self._brackets.append(bracket)

    def _get_new_hyper_parameter_configs(
        self,
        num: int,
        trial_id_prefix: Optional[str] = None
    ):
        if trial_id_prefix is not None:
            trial_id_prefix = "_" + trial_id_prefix
        else:
            trial_id_prefix = ""

        hyper_parameter_configurations = []
        configurations = latin_hypercube_sample(len(self.search_space), num)
        for idx, config in enumerate(configurations):
            config_with_key = {key : config[idx] for idx, key in enumerate(self.search_space)}
            hyper_parameter_configurations.append[
                Trial(
                    trial_id_prefix + str(idx),
                    self.search_space.convert_from_zero_one_scale_to_real_space(config_with_key)
                )
            ]
        return hyper_parameter_configurations


    def get_next_sample(self):
        next_sample = None
        for bracket in self._brackets:
            if not bracket.is_done():
                next_sample = bracket.get_next_sample()
                if self._asynchronous_bracket and next_sample is None:
                    continue
                break

        return next_sample

    def auto_config(self):
        raise NotImplementedError

    def get_progress(self):
        raise NotImplementedError

    def report(self, score, trial_id: str):
        bracket_idx = trial_id.split('_')[0]
        self._brackets[int(bracket_idx)].report_score(score, trial_id)

    def is_done(self):
        for bracket in self._brackets:
            if not bracket.is_done():
                return False
        return True
