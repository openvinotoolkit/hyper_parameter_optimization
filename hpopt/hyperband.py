# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""implementation of the hyperband
"""

# pylint: disable=too-many-lines

import json
import os
from math import ceil, log
from typing import Dict, List, Optional, Union

import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction

import hpopt

from .base import HpOpt, Status
from .logger import get_logger
from .utils import dump_as_json

logger = get_logger()


# pylint: disable=too-many-instance-attributes, too-many-branches, too-many-statements
class AsyncHyperBand(HpOpt):
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

    # pylint: disable=too-many-statements
    def __init__(
        self,
        num_brackets: Optional[int] = None,
        min_iterations: Optional[int] = None,
        reduction_factor: int = 2,
        default_hyper_parameters: Optional[Union[Dict, List[Dict]]] = None,
        use_epoch: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if min_iterations is not None:
            if not isinstance(min_iterations, int):
                raise TypeError("min_iterations should be int type")
            if min_iterations < 1:
                raise ValueError(f"min_iterations should be bigger than 1. Your value is {min_iterations}")

        if not isinstance(reduction_factor, int):
            raise TypeError("reduction_factor should be int type")
        if reduction_factor < 1:
            raise ValueError("reduction_factor should be bigger than 1." f" Your value is {reduction_factor}")

        if num_brackets is not None:
            if not isinstance(num_brackets, int):
                raise TypeError("num_brackets should be int type")
            if num_brackets < 1:
                raise ValueError(f"num_brackets should be bigger than 1. Your value is {num_brackets}")

        (
            self._min_iterations,
            self._reduction_factor,
            self._num_brackets,
            self._expected_total_images,
            self._use_epoch,
            self._updatable_schedule,
        ) = (min_iterations, reduction_factor, num_brackets, 0, use_epoch, False)

        self._current_best = {"trial_id": -1, "score": 0, "image": 0}

        # HPO auto configurator
        if self.num_trials is None or self.max_iterations is None or self.subset_ratio is None:
            logger.info("training schedule was not specified for HPO. run auto configuration.")
            self._updatable_schedule = True
            (
                self.num_trials,
                self.max_iterations,
                self.subset_ratio,
                self._expected_total_images,
                self._num_brackets,
                self._min_iterations,
                self.n_imgs_for_min_train,
                self.n_imgs_for_full_train,
            ) = self.auto_config(
                expected_time_ratio=self.expected_time_ratio,
                num_full_epochs=self.num_full_iterations,
                full_dataset_size=self.full_dataset_size,
                subset_ratio=self.subset_ratio,
                non_pure_train_ratio=self.non_pure_train_ratio,
                num_hyperparams=len(self.search_space),
                reduction_factor=self._reduction_factor,
                parallelism=self.num_workers,
                min_epochs=self._min_iterations,
                max_epochs=self.max_iterations,
            )
            logger.debug(
                f"AsyncHyperBand.__init__() num_trials : {self.num_trials}, min_iterations : {self._min_iterations}, "
                f"max_iterations : {self.max_iterations}, subset_ratio {self.subset_ratio}, "
                f"n_imgs_for_min_train: {self.n_imgs_for_min_train}, "
                f"n_imgs_for_full_train: {self.n_imgs_for_full_train}, "
                f"expected_total_images : {self._expected_total_images}, num_brackets: {self._num_brackets}"
            )
        else:
            if self.max_iterations is None:
                self.max_iterations = self.num_full_iterations

            if self._num_brackets is None:
                self._num_brackets = int(log(self.max_iterations) / log(self._reduction_factor)) + 1

            if self._min_iterations is None:
                self._min_iterations = 1

            if self.subset_ratio is None:
                self.subset_ratio = 1.0

            self.n_imgs_for_full_train = self.full_dataset_size * self.subset_ratio * self.max_iterations
            self.n_imgs_for_min_train = self.full_dataset_size * self.subset_ratio * self._min_iterations

            if self._min_iterations > self.max_iterations:
                logger.info("min_epochs should be less than or equal to max_epochs.")
                self._min_iterations = self.max_iterations

        # Initialize a bayesopt optimizer.
        self._init_bayse_opt(self.search_space)

        # All information in self.hpo_status will be stored at hpo_file_path.
        hpo_file_path = hpopt.get_status_path(self.save_path)

        if self.resume is True and os.path.exists(hpo_file_path):
            with open(hpo_file_path, "rt", encoding="utf-8") as json_file:
                self.hpo_status = json.load(json_file)
                json_file.close()
            self._verify_hpo_status()
        else:
            self.hpo_status["search_algorithm"] = "asha"
            self.hpo_status["search_space"] = {ss: self.search_space[ss].__dict__ for ss in self.search_space}
            self.hpo_status["metric"] = self.metric
            self.hpo_status["subset_ratio"] = self.subset_ratio
            self.hpo_status["image_resize"] = self.image_resize
            self.hpo_status["full_dataset_size"] = self.full_dataset_size
            self.hpo_status["reduction_factor"] = self._reduction_factor
            self.hpo_status["config_list"] = []

        # num_gen_config represents the number of assigned configs.
        # If 1 configs are done and 2 configs are running, num_gen_config is 3.
        self.hpo_status["num_gen_config"] = 0

        # Initialize the brackets
        n0_in_brackets, num_ready_configs = self._init_brackets(default_hyper_parameters)

        # Generate trial configs up to self.num_trials
        if isinstance(self.num_trials, int):
            if self.num_trials > num_ready_configs:
                self._generate_configs(num_ready_configs, self.num_trials, n0_in_brackets)

        # Sync the current status to the file system.
        self.save_results()

    def _init_bayse_opt(self, search_space):
        """initialize bayes opt. it will be used for generating trials."""
        bayesopt_space = {}
        for param in search_space:
            bayesopt_space[param] = (
                search_space[param].lower_space(),
                search_space[param].upper_space(),
            )

        self.optimizer = BayesianOptimization(
            f=self.obj,
            pbounds=bayesopt_space,
            verbose=self.verbose,
            random_state=None,
        )

        self.utilty_function = UtilityFunction(
            kind="ucb",
            kappa=2.5,
            xi=0.0,
            kappa_decay=1,
            kappa_decay_delay=self.num_init_trials,
        )

        if self.has_categorical_param(self.search_space):
            self.optimizer.set_gp_params(alpha=1e-3)

    def _verify_hpo_status(self):
        """verify current hpo_status"""
        if self.hpo_status["search_algorithm"] != "asha":
            logger.error("Search algorithm is changed. Stop resuming.")
            raise ValueError("Search algorithm is changed.")

        if self.hpo_status["search_space"] != {ss: self.search_space[ss].__dict__ for ss in self.search_space}:
            logger.error("Search space is changed. Stop resuming.")
            raise ValueError("Search space is changed.")

        if self.hpo_status["metric"] != self.metric:
            logger.error("Metric is changed. Stop resuming.")
            raise ValueError("Metric is changed.")

        if self.hpo_status["subset_ratio"] != self.subset_ratio:
            logger.error("subset_ratio is changed. Stop resuming.")
            raise ValueError("subset_ratio is changed.")

        if self.hpo_status["image_resize"] != self.image_resize:
            logger.error("image_resize is changed. Stop resuming.")
            raise ValueError("image_resize is changed.")

        if self.hpo_status["full_dataset_size"] != self.full_dataset_size:
            logger.error("The size of dataset is changed. Stop resuming.")
            raise ValueError("Dataset is changed.")

        if self.hpo_status["reduction_factor"] != self._reduction_factor:
            logger.error("reduction_factor is changed. Stop resuming.")
            raise ValueError("reduction_factor is changed.")

        # verify config_list
        for idx, config in enumerate(self.hpo_status["config_list"]):
            if config["status"] == Status.STOP:
                # This case means that config is done.
                self.hpo_status["num_gen_config"] += 1
            else:
                # This case means that config should be in the ready state.
                config["status"] = Status.READY
                trial_file_path = hpopt.get_trial_path(self.save_path, idx)
                if os.path.exists(trial_file_path):
                    # Incomplete informations should be removed before starting.
                    os.remove(trial_file_path)

    def _init_brackets(self, default_hyper_parameters):
        self.rungs_in_brackets = []
        for slot in range(self._num_brackets):
            self.rungs_in_brackets.append(
                self.get_rungs(
                    self.n_imgs_for_min_train,
                    self.n_imgs_for_full_train,
                    self._reduction_factor,
                    slot,
                )
            )

        # Get the max rung iterations
        # self.max_rung = 1
        # for rungs in self.rungs_in_brackets:
        #    curr_max_rung = max(rungs)
        #    if curr_max_rung > self.max_rung:
        #        self.max_rung = curr_max_rung

        # Set max iterations to max rung iterations
        # self.max_iterations = self.max_rung

        # n0 is come from the notation in the paper, which means
        # the initial number of trials in each bracket.
        n0_in_brackets = self.get_num_trials_in_brackets(self._reduction_factor, self._num_brackets)

        # Assign each trial to a bracket
        # according to the ratio of the relative number of trials in each bracket
        brackets_total = sum(n0_in_brackets)
        brackets_ratio = [float(b / brackets_total) for b in n0_in_brackets]

        if default_hyper_parameters is not None:
            if isinstance(default_hyper_parameters, dict):
                default_hyper_parameters = [default_hyper_parameters]

            for idx, default_hyper_parameter in enumerate(default_hyper_parameters):
                self.hpo_status["config_list"].append(
                    {
                        "trial_id": idx,
                        "config": default_hyper_parameter,
                        "status": Status.READY,
                        "score": None,
                        "bracket": 0,
                    }
                )

        num_ready_configs = len(self.hpo_status["config_list"])

        for i, _ in enumerate(n0_in_brackets):
            n0_in_brackets[i] = int(brackets_ratio[i] * self.num_trials)

        n0_in_brackets[0] += self.num_trials - sum(n0_in_brackets)

        for trial in self.hpo_status["config_list"]:
            n0_in_brackets[trial["bracket"]] -= 1

        return n0_in_brackets, num_ready_configs

    def _generate_configs(self, num_ready_configs, num_trials, n0_in_brackets):
        for i in range(num_ready_configs, num_trials):
            bracket_id = 0
            for idx in reversed(range(len(n0_in_brackets))):
                remained_num = n0_in_brackets[idx]
                if remained_num > 0:
                    bracket_id = idx
                    n0_in_brackets[idx] -= 1
                    break

            # Generate a new config
            duplicated = True
            retry_count = 0

            # Check if the new config is duplicated
            while duplicated is True:
                config = self.get_real_config(self.optimizer.suggest(self.utilty_function))
                duplicated = self.check_duplicated_config(config)
                retry_count += 1

                if retry_count > 100:
                    return None

            self.hpo_status["config_list"].append(
                {
                    "trial_id": i,
                    "config": config,
                    "status": Status.READY,
                    "score": None,
                    "bracket": bracket_id,
                }
            )

    @staticmethod
    def get_rungs(min_t: int, max_t: int, reduction_factor: int, slot: int):
        """Get the list of iteration numbers for rungs in the bracket"""
        max_rungs = int(log(max_t - min_t + 1) / log(reduction_factor) - slot + 1)
        rungs = [(min_t - 1) + reduction_factor ** (k + slot) for k in reversed(range(max_rungs))]

        return rungs

    @staticmethod
    def get_num_trials_in_brackets(reduction_factor: int, num_brackets: int):
        """calculate available number of trials for each bracket"""
        brackets = []

        for slot in reversed(range(num_brackets)):
            num = int(ceil(int(num_brackets / (slot + 1)) * reduction_factor**slot))
            brackets.append(num)

        return brackets

    def save_results(self):
        hpo_file_path = hpopt.get_status_path(self.save_path)
        dump_as_json(hpo_file_path, self.hpo_status)

    def get_next_sample(self):
        # Gather all results from workers
        self.update_scores()

        # Check if unassigned configs
        if self.hpo_status["num_gen_config"] >= self.num_trials:
            return None

        # Check total number of trained images
        if self._expected_total_images > 0:
            num_trained_images = self.get_num_trained_images()
            logger.debug(
                f"expected total images = {self._expected_total_images} "
                f"num_trained_image = {num_trained_images}"
                f"number of image best trial used to trian = {self._current_best['image']}"
            )

            if (self._expected_total_images * 1.2
                < num_trained_images + self._current_best['image']
            ):
                return None

        # Choose a config
        next_config = None
        trial_id = None
        for idx, config in enumerate(self.hpo_status["config_list"]):
            if config["status"] is Status.READY:
                config["status"] = Status.RUNNING
                next_config = config
                trial_id = idx
                break

        if next_config is None:
            return None

        self.hpo_status["num_gen_config"] += 1
        self.save_results()

        new_config = {}
        new_config["params"] = next_config["config"]
        new_config["iterations"] = self.max_iterations
        new_config["trial_id"] = trial_id
        new_config["bracket"] = next_config["bracket"]
        new_config["rungs"] = self.rungs_in_brackets[next_config["bracket"]]
        new_config["file_path"] = hpopt.get_trial_path(self.save_path, new_config["trial_id"])
        new_config["subset_ratio"] = self.subset_ratio
        new_config["resize_width"] = self.image_resize[0]
        new_config["resize_height"] = self.image_resize[1]
        # added to calculate number of iterations
        new_config["dataset_size"] = self.full_dataset_size
        new_config["mode"] = self.mode
        new_config["iteration_limit"] = self.n_imgs_for_full_train
        new_config["batch_size_param_name"] = self.batch_size_name

        return new_config

    def update_scores(self):
        is_score_update = False
        for trial_id, config_item in enumerate(self.hpo_status["config_list"]):
            if config_item["status"] == Status.RUNNING:
                current_status = hpopt.get_current_status(self.save_path, trial_id)

                if current_status == Status.STOP:
                    # score = hpopt.get_best_score(self.save_path, trial_id, self.mode)
                    score, n_imgs = hpopt.get_best_score_with_num_imgs(self.save_path, trial_id, self.mode)

                    if score is not None:
                        config_item["score"] = score
                        config_item["status"] = Status.STOP
                        real_config = config_item["config"]
                        logger.info(f"#{trial_id} | {real_config} | {score}")
                        if (self.mode == "max" and score > self._current_best["score"]) or (
                            self.mode == "min" and score < self._current_best["score"]
                        ):
                            self._current_best["score"] = score
                            self._current_best["trial_id"] = trial_id
                            self._current_best["image"] = n_imgs
                            is_score_update = True
                elif current_status == Status.CUDAOOM:
                    config_item["status"] = Status.READY
                    self.hpo_status["num_gen_config"] -= 1

                    trial_file_path = hpopt.get_trial_path(self.save_path, trial_id)
                    if os.path.exists(trial_file_path):
                        # Failed information should be removed before starting.
                        os.remove(trial_file_path)

                    # shrink search space of 'bs'
                    self.shrink_bs_search_space(config_item)
            # elif config_time["status"] == hpopt.Status.STOP:
            #     score, n_imgs = hpopt.get_best_score_with_num_imgs(self.save_path, trial_id, self.mode)
            #     if score > current_best_score:
            #         current_best_score = score
            #         current_best_trial_id = trial_id
            #         current_best_images = n_imgs

        self.save_results()

        # If the current schedule is updatable, adjust it regarding to the latest information
        if self._updatable_schedule and is_score_update:
            logger.info("best trial changed. updating schedule...")
            _, new_expected_total_images = self._calc_total_budget(
                self._num_brackets,
                self.expected_time_ratio,
                self._reduction_factor,
                self.n_imgs_for_min_train,
                self._current_best["image"],
                self.subset_ratio,
                self.non_pure_train_ratio,
                self.num_workers,
            )
            logger.info(
                f"updated expected total images from {self._expected_total_images} " f"to {new_expected_total_images}"
            )
            self._expected_total_images = new_expected_total_images

    def shrink_bs_search_space(self, not_allowed_config):
        """Lower the upper bound of batch size"""
        # Stop if batch_size_name is not specified.
        if self.batch_size_name is None:
            return

        # Check if batch-size is in the list of tuning params.
        if self.batch_size_name not in self.search_space:
            return

        # Check if the search space type for batch size if qloguniform or quniform.
        if self.search_space[self.batch_size_name].type not in [
            "qloguniform",
            "quniform",
        ]:
            return

        not_allowed_bs = not_allowed_config["config"][self.batch_size_name]
        new_upper_bound = not_allowed_bs - self.search_space[self.batch_size_name].range[2]

        # if new_upper_bound is greater than the current upper bound, update only this trial.
        if self.search_space[self.batch_size_name].range[1] <= new_upper_bound:
            not_allowed_config["config"] = self.get_real_config(self.optimizer.suggest(self.utilty_function))
            return

        self.search_space[self.batch_size_name].range[1] = new_upper_bound

        # if the new upper bound is less than the current lower bound,
        # update the lower bound to be half of the new upper bound.
        if self.search_space[self.batch_size_name].range[0] > self.search_space[self.batch_size_name].range[1]:
            new_lower_bound = self.search_space[self.batch_size_name].range[1] // 2
            self.search_space[self.batch_size_name].range[0] = max(new_lower_bound, 2)

            if self.search_space[self.batch_size_name].range[0] > self.search_space[self.batch_size_name].range[1]:
                raise ValueError("This model cannot be trained even with batch size of 2.")

        # Reset bayes opt with updated search space
        self._init_bayse_opt(self.search_space)

        for config in self.hpo_status["config_list"]:
            if config["status"] is Status.READY:
                config["config"] = self.get_real_config(self.optimizer.suggest(self.utilty_function))

    def get_num_trained_images(self):
        """get total number of images that were used for training"""
        num_trained_images = 0

        for trial_id, _ in enumerate(self.hpo_status["config_list"]):
            trial_file_path = hpopt.get_trial_path(self.save_path, trial_id)
            trial_results = hpopt.load_json(trial_file_path)
            if trial_results is not None:
                images = trial_results.get("images", None)
                if images is not None:
                    num_trained_images += images[-1]

        return num_trained_images

    # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
    def auto_config(
        self,
        expected_time_ratio: Union[int, float],
        num_full_epochs: int,
        full_dataset_size: Optional[int],
        subset_ratio: Optional[Union[float, int]],
        non_pure_train_ratio: float,
        num_hyperparams: int,
        reduction_factor: int,
        parallelism: int,
        min_epochs: Optional[int],
        max_epochs: Optional[int],
    ):
        """generate configuration automatically"""
        # All arguments should be specified.
        if expected_time_ratio is None:
            raise ValueError("expected_time_ratio should be specified.")

        # Apply margin to expected_time_ratio
        # The exact execution time ASHA is fundametally unpredictable.
        # 0.8 is an empirically obtained value.
        expected_time_ratio = expected_time_ratio * 0.8

        if num_full_epochs is None:
            raise ValueError("num_full_epochs should be specified.")

        if non_pure_train_ratio is None:
            raise ValueError("non_pure_train_ratio should be specified.")

        if num_hyperparams is None:
            raise ValueError("num_hyperparams should be specified.")

        if reduction_factor is None:
            reduction_factor = 2

        logger.debug(
            f"called auto_config(expected_time_ratio = {expected_time_ratio}, "
            f"num_full_epochs = {num_full_epochs}, "
            f"full_dataset_size = {full_dataset_size}, "
            f"subset_ratio = {subset_ratio}, "
            f"non_pure_train_ratio = {non_pure_train_ratio}, "
            f"num_hyperparams = {num_hyperparams}, "
            f"reduction_factor = {reduction_factor}, "
            f"parallelism = {parallelism}, "
            f"min_epochs = {min_epochs}, "
            f"max_epochs = {max_epochs}, "
        )

        # Create the rung schedule

        # 1. Decide the max epochs
        #    Check if there are early stopped trials
        # if True:
        # set the max epochs to the min. epochs of them
        # else
        # Check the max. epochs in the results
        # set the max epochs to the next rung schedule larger than the number

        # 1. Config target parameters as large as possible
        # max_iterations = -1
        max_epochs = num_full_epochs if max_epochs is None else max_epochs

        min_epochs = 1 if min_epochs is None else min_epochs

        min_epochs = min(min_epochs, max_epochs)

        if not isinstance(full_dataset_size, int):
            raise TypeError(f"full dataset size should be integer type but {type(full_dataset_size)}")

        if subset_ratio is None:
            # Default subset ratio is 0.2
            subset_ratio = 0.2

            if full_dataset_size == 0:
                logger.warning("Sub-dataset isn't used because full_dataset_size value is 0.")
                subset_ratio = 1.0
            elif (full_dataset_size * subset_ratio) < self.min_subset_size:
                if full_dataset_size > self.min_subset_size:
                    subset_ratio = float(self.min_subset_size) / full_dataset_size
                else:
                    subset_ratio = 1.0

        n_imgs_for_full_train = int(full_dataset_size * max_epochs * subset_ratio)
        n_imgs_for_min_train = int(full_dataset_size * min_epochs * subset_ratio)
        logger.debug(
            f"n_imgs_for_full_train = {n_imgs_for_full_train}, " f"n_imgs_for_min_train = {n_imgs_for_min_train}"
        )

        # num_trials = 0
        # num_brackets = 1
        # current_time_ratio = expected_time_ratio * 0.5

        # Update num_full_images from previous trials
        max_num_images_in_trials = 0

        if self.hpo_status.get("config_list", None) is not None:
            for trial_id, _ in enumerate(self.hpo_status["config_list"]):
                trial_results = hpopt.get_trial_results(self.save_path, trial_id)
                if trial_results is not None:
                    n_images = trial_results.get("images", None)
                    if n_images is not None:
                        if n_images[-1] > max_num_images_in_trials:
                            max_num_images_in_trials = n_images[-1]

            logger.debug(f"(before) max_num_images_in_trials: {max_num_images_in_trials}")

            if max_num_images_in_trials > 0:
                rungs = self.get_rungs(
                    max_t=n_imgs_for_full_train,
                    min_t=n_imgs_for_min_train,
                    reduction_factor=reduction_factor,
                    slot=0,
                )

                if len(rungs) > 5:
                    rungs = rungs[:-5]
                else:
                    rungs = rungs[:1]

                logger.debug(f"rungs: {rungs}")
                for rung in reversed(rungs):
                    if rung > max_num_images_in_trials:
                        n_imgs_for_full_train = rung
                        break
            logger.debug(f"(after) new n_imgs_for_full_train:{n_imgs_for_full_train}")

        # 2. Update the target parameters iteratively
        num_brackets = 1
        num_trials, num_total_images = self._calc_total_budget(
            num_brackets,
            expected_time_ratio,
            reduction_factor,
            n_imgs_for_min_train,
            n_imgs_for_full_train,
            subset_ratio,
            non_pure_train_ratio,
            parallelism,
        )

        logger.debug(
            f"auto_config() results: num_trials ={num_trials}, "
            f"max_epochs = {max_epochs}, min_epochs = {min_epochs}, "
            f"n_imgs_for_full_train {n_imgs_for_full_train}, "
            f"n_imgs_for_min_train = {n_imgs_for_min_train}, "
            f"num_total_images = {num_total_images}, "
            f"num_brackets = {num_brackets}, subset_ratio = {subset_ratio}"
        )

        return (
            num_trials,
            max_epochs,
            subset_ratio,
            num_total_images,
            num_brackets,
            min_epochs,
            n_imgs_for_min_train,
            n_imgs_for_full_train,
        )

    # pylint: disable=too-many-arguments
    def _calc_total_budget(
        self,
        num_brackets,
        time_ratio,
        reduction_factor,
        resource_min,
        resource_max,
        subset_ratio,
        non_trainset_ratio,
        parallelism,
    ):
        num_trials = 0
        current_time_ratio = time_ratio * 0.5
        while current_time_ratio < time_ratio:
            num_trials = num_trials + 1
            num_total_images = self.get_total_n_images(
                num_trials, reduction_factor, num_brackets, resource_min, resource_max
            )

            current_time_ratio = (
                num_total_images
                / resource_max
                / parallelism
                * ((1 - non_trainset_ratio) * subset_ratio + non_trainset_ratio)
            )
        return num_trials, num_total_images

    def get_total_n_images(
        self,
        num_trials: int,
        reduction_factor: int,
        num_brackets: int,
        n_imgs_min_train: int,
        n_imgs_full_train: int,
    ):
        """calculate total number of images for given number of trials"""
        num_total_images = 0

        num_trials_in_brackets = self.get_num_trials_in_brackets(reduction_factor, num_brackets)

        brackets_ratio = [float(b / sum(num_trials_in_brackets)) for b in num_trials_in_brackets]

        for i, _ in enumerate(num_trials_in_brackets):
            num_trials_in_brackets[i] = int(brackets_ratio[i] * num_trials)

        num_trials_in_brackets[0] += num_trials - sum(num_trials_in_brackets)

        for i, n_trials in enumerate(num_trials_in_brackets):
            rungs = self.get_rungs(n_imgs_min_train, n_imgs_full_train, reduction_factor, i)
            remained_trials = n_trials
            for rung in reversed(rungs):
                num_total_images += (remained_trials - (remained_trials // reduction_factor)) * rung
                remained_trials = remained_trials // reduction_factor
            num_total_images += remained_trials * n_imgs_full_train

        return num_total_images

    def get_progress(self):
        # num images based progress
        image_progress = min(self.get_num_trained_images() / self._expected_total_images, 0.99)
        # trial based progress
        finished_trials = sum([val["status"] == Status.STOP for val in self.hpo_status["config_list"]])
        trial_progress = finished_trials / self.num_trials

        # return min(0.99, max(epoch_progress, trial_progress))
        logger.debug(
            f"get_progress() iter = {image_progress}/{self._expected_total_images}"
            f", trial {trial_progress}/{self.num_trials}"
        )

        return min(0.99, max(image_progress, trial_progress))

    @staticmethod
    def report(
        config,
        trial_results,
    ):
        """report implementation for the hyperband"""
        if trial_results["images"][-1] >= config["iteration_limit"]:
            trial_results["status"] = Status.STOP
            logger.info("trial reaches its limitation. stop")
            return

        # Async HyperBand
        save_path = os.path.dirname(config["file_path"])
        # curr_itr = len(trial_results['scores'])
        curr_itr = trial_results["images"][-1]
        logger.debug(f"current iterations = {curr_itr} : {config['rungs']}")

        for rung_itr in config["rungs"]:
            if curr_itr >= rung_itr:
                # Decide whether to promote to the next rung or not
                logger.debug(f"{save_path} / curr_itr = {curr_itr} / rung_itr = {rung_itr}")
                cutoff_score = _get_cutoff_score(save_path, rung_itr, config["rungs"], config["mode"])

                if cutoff_score is not None:
                    if config["mode"] == "min":
                        curr_best_score = min(trial_results["scores"])
                    else:
                        curr_best_score = max(trial_results["scores"])

                    stop_flag = False

                    if config["mode"] == "max" and cutoff_score > curr_best_score:
                        stop_flag = True
                    elif config["mode"] == "min" and cutoff_score < curr_best_score:
                        stop_flag = True

                    if stop_flag:
                        trial_results["status"] = Status.STOP
                        logger.debug(
                            f"[ASHA STOP] [{config['trial_id']}, {curr_itr}, {rung_itr}] "
                            f"{cutoff_score} > {curr_best_score}"
                        )
                        break


def _get_percentile(_list: List[int], percentile: float, mode: str):
    if mode == "max":
        percentile = np.nanpercentile(_list, (1 - percentile) * 100)
    else:
        percentile = np.nanpercentile(_list, percentile * 100)
    logger.debug(f"get {mode} percentile of {percentile} from {_list}, ret = {percentile}")
    return percentile


def _get_all_results(save_path):
    if not os.path.exists(hpopt.get_status_path(save_path)):
        logger.warning(f"not existed status json file {save_path}")
        return None, None, None

    hpo_status = hpopt.load_json(hpopt.get_status_path(save_path))

    if hpo_status is None:
        logger.warning(f"failed to load hpo status json file in {save_path}")
        return None, None, None

    hpo_trial_results_scores = []
    hpo_trial_results_imgs_num = []

    for idx, config in enumerate(hpo_status["config_list"]):
        if config["status"] in [Status.RUNNING, Status.STOP]:
            trial_results = hpopt.get_trial_results(save_path, idx)
            if len(trial_results["scores"]) != len(trial_results["images"]):
                raise RuntimeError(
                    f"mismatch length of trial results and number of trained images {len(trial_results['scores'])}"
                    f"/{len(trial_results['images'])}"
                )
            if trial_results is not None:
                hpo_trial_results_scores.append(trial_results["scores"])
                hpo_trial_results_imgs_num.append(trial_results["images"])
    return (
        hpo_trial_results_scores,
        hpo_trial_results_imgs_num,
        hpo_status["reduction_factor"],
    )


def _find_idx_of_max_less_than_target(src: list, target: Union[int, float]):
    idx = -1
    # TODO need to sort list?
    if len(src) <= 0:
        logger.warning("src list is empty")
    elif src[0] > target:
        logger.info("target is out of range (lower bound)")
    elif src[-1] < target:
        logger.info("target is out of range (upper bound)")
    else:
        idx = src.index(max(x for x in src if x <= target))
    logger.debug(f"found index is {idx} which is minimum " f"but greater than {target} from src = {src}")
    return idx


# pylint: disable=too-many-nested-blocks
def _get_cutoff_score(save_path: str, target_rung: int, _rung_list, mode: str):
    """
    Calculate a cutoff score for a specified rung of ASHA

    Args:
        save_path (str): path where result of HPO is saved.
        target_rung (int): current rung number
        _rung_list (list): list of rungs in the current bracket
        mode (str): max or min. Decide whether to find max value or min value.
    """
    logger.debug(f"get_cutoff_score({target_rung}, {_rung_list}, {mode})")

    # Gather all scores with iter number
    (
        hpo_trial_results_scores,
        hpo_trial_results_imgs_num,
        reduction_factor,
    ) = _get_all_results(save_path)
    if hpo_trial_results_imgs_num is None or hpo_trial_results_scores is None:
        return None

    logger.debug(f"scores = {hpo_trial_results_scores} num_imgs = {hpo_trial_results_imgs_num}")

    # Run a SHA (not ASHA)
    rung_score_list: List[int] = []
    rung_list = _rung_list.copy()
    rung_list.sort(reverse=False)

    if len(hpo_trial_results_scores) == 1:
        return None

    for curr_rung in rung_list:
        if curr_rung <= target_rung:
            rung_score_list.clear()
            logger.debug(f"curr_rung = {curr_rung}")
            for idx, num_imgs in enumerate(hpo_trial_results_imgs_num):
                if num_imgs != []:
                    logger.debug(f"trial {idx}: score reported @ {num_imgs}")
                    logger.debug(f"trial {idx}: scores {hpo_trial_results_scores[idx]}")
                    result_idx = _find_idx_of_max_less_than_target(num_imgs, curr_rung)
                    logger.debug(f"cutoff idx = {result_idx}")
                    if result_idx != -1:
                        if result_idx == 0:
                            rung_score_list.append(hpo_trial_results_scores[idx][result_idx])
                        else:
                            if mode == "max":
                                rung_score_list.append(max(hpo_trial_results_scores[idx][:result_idx]))
                            else:
                                rung_score_list.append(min(hpo_trial_results_scores[idx][:result_idx]))
                else:
                    logger.debug(f"trial {idx}: skipped since num_imgs is empty")
                    num_imgs.clear()
                    hpo_trial_results_scores[idx].clear()

            if len(rung_score_list) > 1:
                logger.debug(f"rung_score_list = {rung_score_list}")
                cut_off_score = _get_percentile(rung_score_list, 1 / reduction_factor, mode)
                logger.debug(f"cut_off_score = {cut_off_score}")
                for idx, num_imgs in enumerate(hpo_trial_results_imgs_num):
                    if num_imgs != []:
                        result_idx = _find_idx_of_max_less_than_target(num_imgs, curr_rung)
                        if result_idx != -1:
                            if result_idx == 0:
                                target_score = hpo_trial_results_scores[idx][result_idx]
                            else:
                                target_score = (
                                    max(hpo_trial_results_scores[idx][:result_idx])
                                    if mode == "max"
                                    else min(hpo_trial_results_scores[idx][:result_idx])
                                )
                            if mode == "max":
                                if target_score < cut_off_score:
                                    hpo_trial_results_scores[idx].clear()
                                    num_imgs.clear()
                                    logger.debug(f"removed results from trial {idx}")
                            else:
                                if target_score > cut_off_score:
                                    hpo_trial_results_scores[idx].clear()
                                    num_imgs.clear()
                                    logger.debug(f"removed results from trial {idx}")
    if len(rung_score_list) > 1:
        logger.debug(f"last rung_score_list = {rung_score_list}")
        return _get_percentile(rung_score_list, 1 / reduction_factor, mode)
    return None
