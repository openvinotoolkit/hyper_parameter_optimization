# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
""" implementation of the base module
"""

import math

# import logging
import time
from enum import IntEnum
from typing import List, Optional, Union

from .logger import get_logger
from .utils import type_check

logger = get_logger()


class Status(IntEnum):
    """
    HPO status enum
    """

    UNKNOWN = -1
    READY = 0
    RUNNING = 1
    STOP = 2
    CUDAOOM = 3
    # Previous HPO task is not found.
    NORESULT = 4
    # Previous HPO task is started but not finished yet.
    PARTIALRESULT = 5
    # Previous HPO task is completely finished. It means that the best hyper-parameters are already found.``
    COMPLETERESULT = 6


class SearchSpace:
    """
    This implements search space used for SMBO and ASHA.
    This class support uniform and quantized uniform with normal and log scale
    in addition to categorical type. Quantized type has step which is unit for change.

    Args:
        type (str): type of hyper parameter in search space.
                    supported types: uniform, loguniform, quniform, qloguniform, choice
        range (list): range of hyper parameter search space.
                      What value at each position means is as bellow.
                      uniform: [lower space, upper space]
                      quniform: [lower space, upper space, step]
                      loguniform: [lower space, upper space, logarithm base]
                      qloguniform: [lower space, upper space, step, logarithm base]
                      categorical: [each categorical values, ...]
    """

    def __init__(self, _type: str, _range: List[Union[float, int]]):
        self._type = _type
        self._range = _range

        if self._type in ("uniform", "loguniform"):
            if len(self._range) == 2:
                self._range.append(2)
            elif len(self._range) < 2:
                raise ValueError(
                    f"The range of the {self._type} type requires "
                    "two numbers for lower and upper limits. "
                    f"Your value is {self._range}"
                )
        elif self._type in ("quniform", "qloguniform"):
            if len(self._range) == 3:
                self._range.append(2)
            elif len(self._range) < 3:
                raise ValueError(
                    f"The range of the {self._type} type requires "
                    "three numbers for lower/upper limits and "
                    "quantization number. "
                    f"Your value is {self._range}"
                )
        elif self._type == "choice":
            self._range = [0, len(_range)]
            self.choice_list = _range
        else:
            raise TypeError(f"{self._type} is an unknown search space type.")

    def __repr__(self):
        return f"type: {self._type}, range: {self._range}"

    def lower_space(self):
        """get lower bound from range"""
        if self._type == "loguniform":
            return math.log(self._range[0], self._range[2])
        if self._type == "qloguniform":
            return math.log(self._range[0], self._range[3])

        return self._range[0]

    def upper_space(self):
        """get upper bound from range"""
        if self._type == "loguniform":
            return math.log(self._range[1], self._range[2])
        if self._type == "qloguniform":
            return math.log(self._range[1], self._range[3])

        return self._range[1]

    def space_to_real(self, number: Union[int, float]):
        """convert method"""
        if self._type == "quniform":
            return round(number / self._range[2]) * self._range[2]
        if self._type == "loguniform":
            return self._range[2] ** number
        if self._type == "qloguniform":
            return round(self._range[3] ** number / self._range[2]) * self._range[2]
        if self._type == "choice":
            idx = int(number)
            idx = min(idx, len(self.choice_list) - 1)
            idx = max(idx, 0)
            return self.choice_list[idx]

        return number

    def real_to_space(self, number: Union[int, float]):
        """convert method"""
        if self._type == "loguniform":
            return math.log(number, self._range[2])
        if self._type == "qloguniform":
            return math.log(number, self._range[3])

        return number

    @property
    def type(self):
        """property method for the internal attribute"""
        return self._type

    @property
    def range(self):
        """property method for the internal attribute"""
        return self._range


# pylint: disable=too-many-instance-attributes
class HpOpt:
    """
    This implements class which make frame for bayesian optimization
    or ahsa class. So, only common methods are implemented but
    core method for HPO.

    Args:
        save_path (str): path where result of HPO is saved.
        search_space (dict): hyper parameter search space to find.
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
        max_time: TBD
        resources_per_trial: TBD
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

    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def __init__(
        self,
        search_space: dict,
        save_path: str = "/tmp/hpopt",
        mode: str = "max",
        num_init_trials: int = 5,
        num_trials: Optional[int] = None,
        num_workers: int = 1,
        num_full_iterations: int = 1,
        non_pure_train_ratio: float = 0.2,
        full_dataset_size: int = 0,
        metric: str = "mAP",
        expected_time_ratio: Union[float, int] = 4,
        max_iterations: Optional[int] = None,
        max_time=None,
        resources_per_trial=None,
        subset_ratio: Optional[Union[float, int]] = None,
        min_subset_size: int = 500,
        image_resize: List[int] = None,
        batch_size_name: Optional[str] = None,
        verbose: int = 0,
        resume: bool = False,
    ):
        if save_path is None:
            raise TypeError("save_path should be str or path object, Not NoneType")

        if mode not in ["min", "max"]:
            raise ValueError("'mode' should be one of 'min' or 'max'.")

        type_check(search_space, [dict])

        # if not isinstance(expected_time_ratio, float) and not isinstance(
        #     expected_time_ratio, int
        # ):
        #     raise TypeError("expected_time_ratio should be float or int type.")
        # if expected_time_ratio <= 0:
        #     raise ValueError(
        #         "expected_time_ratio should be bigger than 0."
        #         f" Your value is {expected_time_ratio}"
        #     )
        type_check(expected_time_ratio, [float, int], positive=True, non_zero=True)

        # if not isinstance(full_dataset_size, int):
        #     raise TypeError("full_dataset_size should be int type.")
        # if full_dataset_size < 0:
        #     raise ValueError(
        #         "full_dataset_size should be zero or postive value"
        #         f"Your value is {full_dataset_size}."
        #     )
        type_check(full_dataset_size, int, positive=True)

        # if not isinstance(num_full_iterations, int):
        #     raise TypeError("num_full_iteration should be int type.")
        # if num_full_iterations < 1:
        #     raise ValueError(
        #         "num_full_iterations should be 1 <=."
        #         f" Your value is {num_full_iterations}"
        #     )
        type_check(num_full_iterations, int, positive=True, non_zero=True)

        # if not isinstance(non_pure_train_ratio, float):
        #     raise TypeError("non_pure_train_ratio should be int type.")
        type_check(non_pure_train_ratio, float)
        if 0 > non_pure_train_ratio or 1 < non_pure_train_ratio:
            raise ValueError("non_pure_train_ratio should be between 0 and 1." f" Your value is {non_pure_train_ratio}")

        # if not isinstance(num_init_trials, int):
        #     raise TypeError("num_init_trials should be int type")
        # if num_init_trials < 1:
        #     raise ValueError(
        #         "num_init_trials should be bigger than 0."
        #         f" Your value is {num_init_trials}"
        #     )
        type_check(num_init_trials, int, positive=True, non_zero=True)

        # if max_iterations is not None:
        #     if not isinstance(max_iterations, int):
        #         raise TypeError("max_iterations should be int type")
        #     if max_iterations < 1:
        #         raise ValueError(
        #             "max_iterations should be bigger than 0."
        #             f" Your value is {max_iterations}"
        #         )
        type_check(max_iterations, [int, type(None)], positive=True, non_zero=True)

        # if num_trials is not None:
        #     if not isinstance(num_trials, int):
        #         raise TypeError("num_trials should be int type")
        #     if num_trials < 1:
        #         raise ValueError(
        #             "num_trials should be bigger than 0." f" Your value is {num_trials}"
        #         )
        type_check(num_trials, [int, type(None)], positive=True, non_zero=True)

        # if not isinstance(num_workers, int):
        #     raise TypeError("num_workers should be int type")
        # if num_workers < 1:
        #     raise ValueError(
        #         "num_workers should be bigger than 0." f" Your value is {num_workers}"
        #     )
        type_check(num_workers, int, positive=True, non_zero=True)

        type_check(subset_ratio, [float, int, type(None)])
        if subset_ratio is not None:
            # if not isinstance(subset_ratio, float) and not isinstance(
            #     subset_ratio, int
            # ):
            #     raise TypeError("subset_ratio should be float or int type")
            if 0 >= subset_ratio or 1.0 < subset_ratio:
                raise ValueError("subset_ratio should be > 0 and <= 1." f" Your value is {subset_ratio}")

        if image_resize is None:
            image_resize = [0, 0]

        if not hasattr(image_resize, "__getitem__"):
            raise TypeError("image_resize should be able to accessible by index.")
        if len(image_resize) < 2:
            raise ValueError("image_resize should have at least two values.")
        if image_resize[0] < 0 or image_resize[1] < 0:
            raise ValueError("Each value of image_resize should be positive.")

        if not isinstance(min_subset_size, int):
            raise TypeError("min_subset_size should be an integer value.")

        self.save_path = save_path
        self.search_space = search_space
        self.mode = mode
        self.num_init_trials = num_init_trials
        self.num_trials = num_trials
        self.num_workers = num_workers
        self.num_full_iterations = num_full_iterations
        self.non_pure_train_ratio = non_pure_train_ratio
        self.full_dataset_size = full_dataset_size
        self.expected_time_ratio = expected_time_ratio
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.resources_per_trial = resources_per_trial
        self.subset_ratio = subset_ratio
        self.min_subset_size = min_subset_size
        self.image_resize = image_resize
        self.verbose = verbose
        self.resume = resume
        self.metric = metric
        self.batch_size_name = batch_size_name
        self.hpo_status: dict = {}

    def get_next_sample(self):
        """api definition to get next sample"""

    def get_next_samples(self, num_expected_samples=0):
        """api definition to get next sample"""

    def update_scores(self):
        """api definition to update scores"""

    def save_results(self):
        """api definition to save results"""

    # pylint: disable=unused-argument
    @staticmethod
    def obj(**kwargs):
        """definition of the empty object method for bayes opt"""
        return 0

    @staticmethod
    def has_categorical_param(search_space):
        """helper function to check whether search space config has categorical hp."""
        for param in search_space:
            if search_space[param].type == "choice":
                return True

        return False

    def get_real_config(self, config):
        """convert space to specific value"""
        real_config = {}
        for param in config:
            real_config[param] = self.search_space[param].space_to_real(config[param])
        return real_config

    def get_space_config(self, config):
        """convert specific value to space"""
        space_config = {}
        for param in config:
            space_config[param] = self.search_space[param].real_to_space(config[param])
        return space_config

    def print_results(self):
        """helper function to print out the overall HPO report"""
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
        """helper function to verify the config to avoid redundancy of the trial config"""
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

    def get_best_config(self):
        """api to get the best config from the HPO status"""
        self.update_scores()

        # Wait for updating files up to 5 seconds.
        update_try_count = 0

        while update_try_count < 5:
            update_try_count += 1
            update_flag = False

            for config_item in self.hpo_status["config_list"]:
                if config_item["score"] is None:
                    time.sleep(1)
                    self.update_scores()
                    update_flag = True
                    break

            if not update_flag:
                break

        # Fining the 1st non-null score
        best_score = 0
        best_trial_id = 0

        for trial_id, config_item in enumerate(self.hpo_status["config_list"]):
            if config_item["score"] is not None:
                best_score = config_item["score"]
                best_trial_id = trial_id
                break

        for trial_id, config_item in enumerate(self.hpo_status["config_list"]):
            update_flag = False

            if config_item["score"] is None:
                continue

            if self.mode == "min" and config_item["score"] < best_score:
                update_flag = True
            elif self.mode == "max" and config_item["score"] > best_score:
                update_flag = True

            if update_flag:
                best_score = config_item["score"]
                best_trial_id = trial_id

        self.hpo_status["best_config_id"] = best_trial_id
        self.save_results()

        return self.hpo_status["config_list"][best_trial_id]["config"]

    def get_progress(self):
        """api to get current progress of the HPO process"""
