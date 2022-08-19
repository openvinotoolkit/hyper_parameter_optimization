# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
import time
import os
import json
import subprocess
import pickle
import inspect
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any, Union
from os import path as osp

import hpopt
from hpopt.logger import get_logger
from hpopt.search_space import SearchSpace
from hpopt.utils import check_mode_input, check_positive, check_not_negative

logger = get_logger()


class HpoBase(ABC):
    """
    This implements class which make frame for bayesian optimization
    or ahsa class. So, only common methods are implemented but
    core method for HPO.

    Args:
        save_path (str): path where result of HPO is saved.
        search_space (list): hyper parameter search space to find.
        mode (str): One of {min, max}. Determines whether objective is
                    minimizing or maximizing the metric attribute.
        num_init_trials (int): Only for SMBO. How many trials to use to init SMBO.
        num_trials (int): How many training to conduct for HPO.
        num_workers (int): How many trains are executed in parallel.
        num_full_iterations (int): epoch for traninig after HPO.
        non_pure_train_ratio (float): ratio of validation time to (train time + validation time)
        full_dataset_size (int): train dataset size
        expected_time_ratio (int or float): Time to use for HPO.
                                            If HPO is configured automatically,
                                            HPO use time about exepected_time_ratio *
                                            train time after HPO times.
        max_iterations (int): Max training epoch for each trial.
        subset_ratio (float or int): ratio to how many train dataset to use for each trial.
                                     The lower value is, the faster the speed is.
                                     But If it's too low, HPO can be unstable.
        min_subset_size (float or int) : Minimum size of subset. Default value is 500.
        image_resize (list): Width and height of image used to resize for decreasing HPO time.
                             First value of list is width and second value is height.
        verbose (int): Decide how much content to print.
        resume (bool): resume flag decide to use previous HPO results.
                       If HPO completed, you can just use optimized hyper parameters.
                       If HPO stopped in middle, you can resume in middle.
    """

    def __init__(
        self,
        search_space: Dict[str, Dict[str, Any]],
        save_path: str = "/tmp/hpopt",
        mode: str = "max",
        num_trials: Optional[int] = None,
        num_workers: int = 1,
        num_full_iterations: int = 1,
        non_pure_train_ratio: float = 0.2,
        full_dataset_size: int = 0,
        metric: str = "mAP",
        expected_time_ratio: Union[int, float] = 4,
        maximum_resource: Optional[Union[int, float]] = None,
        subset_ratio: Optional[Union[float, int]] = None,
        min_subset_size=500,
        image_resize: List[int] = [0, 0],
        batch_size_name: Optional[str] = None,
        verbose: int = 0,
        resume: bool = False,
        prior_hyper_parameters: Optional[Union[Dict, List[Dict]]] = None,
    ):
        check_mode_input(mode)
        check_positive(expected_time_ratio, "expected_time_ratio")
        check_positive(full_dataset_size, "full_dataset_size")
        check_positive(num_full_iterations, "num_full_iterations")
        if not (0 < non_pure_train_ratio <= 1):
            raise ValueError(
                "non_pure_train_ratio should be between 0 and 1."
                f" Your value is {non_pure_train_ratio}"
            )
        if maximum_resource is not None:
            check_positive(maximum_resource, "maximum_resource")
        if num_trials is not None:
            check_positive(num_trials, "num_trials")
        check_positive(num_workers, "num_workers")
        if subset_ratio is not None:
            if not (0 < subset_ratio <= 1.0):
                raise ValueError(
                    "subset_ratio should be greater than 0 and lesser than or equal to 1."
                    f" Your value is {subset_ratio}"
                )
        elif len(image_resize) < 2:
            raise ValueError("image_resize should have at least two values.")
        elif image_resize[0] < 0 or image_resize[1] < 0:
            raise ValueError("Each value of image_resize should be positive.")

        self.save_path = save_path
        self.search_space = SearchSpace(search_space)
        self.mode = mode
        self.num_trials = num_trials
        self.num_workers = num_workers
        self.num_full_iterations = num_full_iterations
        self.non_pure_train_ratio = non_pure_train_ratio
        self.full_dataset_size = full_dataset_size
        self.expected_time_ratio = expected_time_ratio
        self.maximum_resource = maximum_resource
        self.subset_ratio = subset_ratio
        self.min_subset_size = min_subset_size
        self.image_resize = image_resize
        self.verbose = verbose
        self.resume = resume
        self.hpo_status: dict = {}
        self.metric = metric
        self.batch_size_name = batch_size_name
        self.prior_hyper_parameters = prior_hyper_parameters

    def save_results(self):
        """Sync the current status to the file system."""
        hpo_file_path = hpopt.get_status_path(self.save_path)
        oldmask = os.umask(0o077)
        with open(hpo_file_path, "wt") as json_file:
            json.dump(self.hpo_status, json_file, indent=4)
            json_file.close()
        os.umask(oldmask)

    def print_results(self):
        field_widths = []
        field_param_name = []
        print(f'|{"#": ^5}|', end="")
        for param in self.search_space:
            field_title = f"{param}"
            filed_width = max(len(field_title) + 2, 20)
            field_widths.append(filed_width)
            field_param_name.append(param)
            print(f"{field_title: ^{filed_width}} |", end="")
        print(f'{"score": ^21}|')

        for trial_id, config_item in enumerate(self.hpo_status["config_list"], start=1):
            if config_item["score"] is not None:
                print(f"|{trial_id: >4} |", end="")
                real_config = config_item["config"]
                for param, field_width in zip(field_param_name, field_widths):
                    print(f"{real_config[param]: >{field_width}} |", end="")
                score = config_item["score"]
                print(f"{score: >20} |", end="")
                print("")

    def check_duplicated_config(self, new_config):
        for old_item in self.hpo_status["config_list"]:
            old_config = old_item["config"]
            matched = True
            for param in old_config:
                old_value = old_config[param]
                new_value = new_config[param]

                if math.isclose(new_value, old_value) is False:
                    matched = False
                    break

            if matched is True:
                return True

        return False

    def run_hpo(self, train_func):
        command = self._get_hpo_runner_command(train_func)
        hpo_loop_process = subprocess.Popen(
            args=command,
            shell=False,
            env=os.environ.copy(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        pickle.dump(self, hpo_loop_process.stdin) # pass hpo alrgorim(self) to subprocess
        hpo_loop_process.wait()
        best_hp = pickle.load(hpo_loop_process.stdout)

        return best_hp

    def _get_hpo_runner_command(self, train_func):
        module = inspect.getmodule(train_func)
        module_name = module.__name__

        command = ["python3", "-m", "hpopt.hpo_runner"]
        if module_name == "__main__":
            file_path = osp.abspath(module.__file__)
            module_name = osp.basename(file_path).split('.')[0]
            command.extend(
                ["--module-name", module_name, "--file-path", file_path, "--train-function-name", train_func.__name__]
            )
        else:
            command.extend(["--module-name", module_name, "--train-function-name", train_func.__name__])

        return command

    @abstractmethod
    def is_done(self):
        raise NotImplementedError

    @abstractmethod
    def get_next_sample(self):
        raise NotImplementedError

    @abstractmethod
    def auto_config(self):
        raise NotImplementedError

    @abstractmethod
    def get_progress(self):
        raise NotImplementedError

    @abstractmethod
    def report_score(self, score, resource, trial_id):
        raise NotImplementedError

    @abstractmethod
    def get_best_config(self):
        raise NotImplementedError

class Trial:
    def __init__(
        self,
        id: Any,
        configuration: Dict
    ):
        self._id = id
        self._configuration = configuration
        self.score: Dict[Union[float, int], Union[float, int]] = {}

    @property
    def id(self):
        return self._id

    @property
    def configuration(self):
        return self._configuration

    def set_iterations(self, iter: Union[int, float]):
        check_positive(iter, "iter")
        self._configuration["iterations"] = iter

    def register_score(self, score: Union[int, float], resource: Union[int, float]):
        check_positive(resource)
        self.score[resource] = score

    def get_best_score(self, mode: str = "max", resource_limit: Optional[Union[float, int]] = None):
        check_mode_input(mode)

        if resource_limit is None:
            scores = self.score.values()
        else:
            scores = [val for key, val in self.score.items() if key <= resource_limit]

        if len(scores) == 0:
            return None

        if mode == "max":
            return max(scores)
        else:
            return min(scores)

    def get_progress(self):
        if len(self.score) == 0:
            return 0
        return max(self.score.keys())
