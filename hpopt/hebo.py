from typing import Any, Dict, List, Optional, Union
from copy import deepcopy
from os import path as osp
import json

import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import numpy as np

from hpopt.hpo_base import HpoBase, Trial
from hpopt.hpo_base import TrialStatus
from hpopt.logger import get_logger
from hpopt.utils import (
    check_mode_input,
    check_not_negative,
    check_positive,
    left_is_better,
)

logger = get_logger()

class Hebo(HpoBase):
    def __init__(self, **kwargs):
        super(Hebo, self).__init__(**kwargs)
        search_space = self._make_hebo_search_space()
        self._engine = HEBO(search_space)
        self._trials: Dict[str, Trial] = {}
        self._next_trial_id = 0

    def _make_hebo_search_space(self):
        search_space_config = []
        for hp_name, hp in self.search_space.search_space.items():
            lb = hp.lower_space()
            ub = hp.upper_space()
            if (isinstance(lb, int) or lb.is_integer()) and (isinstance(ub, int) or ub.is_integer()):
                space_type = "int"
                lb = int(lb)
                ub = int(ub)
            else:
                space_type = "num"

            search_space_config.append(
                {"name" : hp_name, "type" : space_type, "lb" : lb, "ub" : ub}
            )

        return DesignSpace().parse(search_space_config)

    def save_results(self):
        trials = {}
        for trial_id, trial in self._trials.items():
            trial.save_results(osp.join(self.save_path, f"{trial_id}.json"))
            
            if trial.is_done():
                trials[trial_id] = trial.get_best_score()
            else:
                trials[trial_id] = None

        result = {
            "num_trials" : self.num_trials,
            "max_iteration" : self.maximum_resource,
            "trial" : trials
        }

        with open(osp.join(self.save_path, "hebo.json"), "w") as f:
            json.dump(result, f)

    def is_done(self):
        if len(self._trials) < self.num_trials:
            return False
        
        for trial in self._trials.values():
            if not trial.is_done():
                return False

        return True

    def get_next_sample(self):
        if len(self._trials) >= self.num_trials:
            return None

        if self.prior_hyper_parameters:
            hp = self.prior_hyper_parameters.pop(0)
        else:
            hp_dataframe = self._engine.suggest()
            hp = self._transform_dataframe_to_trial_format(hp_dataframe)

        return self._make_trial(hp)

    def _transform_dataframe_to_trial_format(self, data_frame: pd.DataFrame):
        return {col : self.search_space[col].space_to_real(data_frame[col].values.item()) for col in data_frame.columns}

    def auto_config(self):
        raise NotImplementedError

    def get_progress(self):
        raise NotImplementedError

    def report_score(self, score: Union[float, int], resource: Union[float, int], trial_id: str, done: bool = False):
        trial = self._trials[trial_id]
        if done:
            trial.finalize()
            hp = deepcopy(trial.configuration)
            if "iterations" in hp:
                del hp["iterations"]

            score = trial.get_best_score()
            if self.mode == "max":
                score = -score

            self._engine.observe(
                self._transfrom_to_engine_format(hp),
                self._wrap_scrore_by_ndarray(score)
            )
            return TrialStatus.STOP
        else:
            trial.register_score(score, resource)
            return TrialStatus.RUNNING

    def _wrap_scrore_by_ndarray(self, score: Union[int, float]):
        return np.array([score]).reshape(-1, 1)

    def _transfrom_to_engine_format(self, hp: Dict):
        transformed_hp = {}
        for key, val in hp.items():
            is_int = False
            if isinstance(val, int):
                is_int = True
            val = self.search_space[key].real_to_space(val)
            if is_int:
                val = round(val)

            transformed_hp[key] = [val]

        return pd.DataFrame(transformed_hp)

    def get_best_config(self):
        best_score = None
        best_trial = None

        for trial in self._trials.values():
            score = trial.get_best_score()
            if score is not None and (best_score is None or left_is_better(score, best_score, self.mode)):
                best_score = score
                best_trial = trial

        if best_trial is None:
            return None

        if "iterations" in best_trial.configuration:
            del best_trial.configuration["iterations"]

        return best_trial.configuration

    def print_result(self):
        trials_record=""
        best_score = 0
        best_trial = None
        for trial in self._trials.values():
            score = trial.get_best_score()
            if score is not None and best_score < score:
                best_score = score
                best_trial = trial

            trials_record += f"id : {trial.id} / score : {score} / config : {trial.configuration}\n"

        print(f"best trial => id : {best_trial.id} / score : {best_score} / config : {best_trial.configuration}")
        print(trials_record)

    def _make_trial(self, hyper_parameter: Dict):
        id = self._get_new_trial_id()
        trial = Trial(id, hyper_parameter, self._get_train_environment())
        trial.iteration = self.maximum_resource
        self._trials[id] = trial
        return trial

    def _get_train_environment(self):
        train_environment = {"subset_ratio" : self.subset_ratio}
        return train_environment

    def _get_new_trial_id(self):
        id = self._next_trial_id
        self._next_trial_id += 1
        return str(id)
