# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math

# import logging
import time
from typing import List, Optional, Union

from hpopt.logger import get_logger

logger = get_logger()


class HpOpt:
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

    def __init__(
        self,
        search_space: List,
        save_path: str = "/tmp/hpopt",
        mode: str = "max",
        num_init_trials: int = 5,
        num_trials: Optional[int] = None,
        num_workers: int = 1,
        num_full_iterations: int = 1,
        non_pure_train_ratio: float = 0.2,
        full_dataset_size: int = 0,
        metric: str = "mAP",
        expected_time_ratio: Union[int, float] = 4,
        max_iterations: Optional[int] = None,
        max_time=None,
        resources_per_trial=None,
        subset_ratio: Optional[Union[float, int]] = None,
        min_subset_size=500,
        image_resize: List[int] = [0, 0],
        batch_size_name=None,
        verbose: int = 0,
        resume: bool = False,
    ):

        if mode not in ["min", "max"]:
            raise ValueError("'mode' should be one of 'min' or 'max'.")

        if type(expected_time_ratio) != float and type(expected_time_ratio) != int:
            TypeError("expected_time_ratio should be float or int type.")
        elif expected_time_ratio <= 0:
            ValueError(
                "expected_time_ratio should be bigger than 0."
                f" Your value is {expected_time_ratio}"
            )

        if type(full_dataset_size) != int:
            TypeError("full_dataset_size should be int type.")
        elif full_dataset_size < 0:
            ValueError(
                "full_dataset_size should be postive value"
                f"Your value is {full_dataset_size}."
            )

        if type(num_full_iterations) != int:
            TypeError("num_full_iteration should be int type.")
        elif num_full_iterations < 1:
            raise ValueError(
                "num_full_iterations should be 1 <=."
                f" Your value is {num_full_iterations}"
            )

        if type(non_pure_train_ratio) != float:
            TypeError("non_pure_train_ratio should be int type.")
        elif not (0 < non_pure_train_ratio < 1):
            raise ValueError(
                "non_pure_train_ratio should be between 0 and 1."
                f" Your value is {non_pure_train_ratio}"
            )

        if type(num_init_trials) != int:
            raise TypeError("num_init_trials should be int type")
        elif num_init_trials < 1:
            raise ValueError(
                "num_init_trials should be bigger than 0."
                f" Your value is {num_init_trials}"
            )

        if max_iterations is not None:
            if type(max_iterations) != int:
                raise TypeError("max_iterations should be int type")
            elif max_iterations < 1:
                raise ValueError(
                    "max_iterations should be bigger than 0."
                    f" Your value is {max_iterations}"
                )

        if num_trials is not None:
            if type(num_trials) != int:
                raise TypeError("num_trials should be int type")
            elif num_trials < 1:
                raise ValueError(
                    "num_trials should be bigger than 0." f" Your value is {num_trials}"
                )

        if type(num_workers) != int:
            raise TypeError("num_workers should be int type")
        elif num_workers < 1:
            raise ValueError(
                "num_workers should be bigger than 0." f" Your value is {num_workers}"
            )

        if subset_ratio is not None:
            if type(subset_ratio) != float and type(subset_ratio) != int:
                raise TypeError("subset_ratio should be float or int type")
            elif not (0 < subset_ratio <= 1.0):
                raise ValueError(
                    "subset_ratio should be > 0 and <= 1."
                    f" Your value is {subset_ratio}"
                )

        if not hasattr(image_resize, "__getitem__"):
            raise TypeError("image_resize should be able to accessible by index.")
        elif len(image_resize) < 2:
            raise ValueError("image_resize should have at least two values.")
        elif image_resize[0] < 0 or image_resize[1] < 0:
            raise ValueError("Each value of image_resize should be positive.")

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
        self.hpo_status: dict = {}
        self.metric = metric
        self.batch_size_name = batch_size_name

    def get_next_sample(self):
        pass

    def get_next_samples(self, num_expected_samples=0):
        pass

    def update_scores(self):
        pass

    def save_results(self):
        pass

    def obj(self, **kwargs):
        return 0

    def hasCategoricalParam(self, search_space):
        for param in search_space:
            if search_space[param].type == "choice":
                return True

        return False

    def get_real_config(self, config):
        real_config = {}
        for param in config:
            real_config[param] = self.search_space[param].space_to_real(config[param])
        return real_config

    def get_space_config(self, config):
        space_config = {}
        for param in config:
            space_config[param] = self.search_space[param].real_to_space(config[param])
        return space_config

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

    def get_best_config(self):
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
        pass
