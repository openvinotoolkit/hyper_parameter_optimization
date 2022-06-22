# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
from math import ceil, floor, log
from typing import Dict, List, Optional, Union

from bayes_opt import BayesianOptimization, UtilityFunction

import hpopt
from hpopt.base import HpOpt
from hpopt.logger import get_logger


logger = get_logger()


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
    def __init__(self,
                 num_brackets: Optional[int] = None,
                 min_iterations: Optional[int] = None,
                 reduction_factor: int = 2,
                 default_hyper_parameters: Optional[Union[Dict, List[Dict]]] = None,
                 use_epoch: Optional[bool] = False,
                 **kwargs):
        super(AsyncHyperBand, self).__init__(**kwargs)
        if min_iterations is not None:
            if type(min_iterations) != int:
                raise TypeError('min_iterations should be int type')
            elif min_iterations < 1:
                raise ValueError(f'min_iterations should be bigger than 1. Your value is {min_iterations}')

        if type(reduction_factor) != int:
            raise TypeError('reduction_factor should be int type')
        elif reduction_factor < 1:
            raise ValueError('reduction_factor should be bigger than 1.'
                             f' Your value is {reduction_factor}')

        if num_brackets is not None:
            if type(num_brackets) != int:
                raise TypeError('num_brackets should be int type')
            elif num_brackets < 1:
                raise ValueError(f'num_brackets should be bigger than 1. Your value is {num_brackets}')

        self._min_iterations = min_iterations
        self._reduction_factor = reduction_factor
        self._num_brackets = num_brackets
        self._expected_total_images = 0
        self._use_epoch = use_epoch
        self._updatable_schedule = False

        # HPO auto configurator
        if self.num_trials is None or self.max_iterations is None or self.subset_ratio is None:
            logger.info("training schedule was not specified for HPO. run auto configuration.")
            self._updatable_schedule = True
            self.num_trials, self.max_iterations, self.subset_ratio, \
                self._expected_total_images, self._num_brackets, self._min_iterations, \
                self.n_imgs_for_min_train, self.n_imgs_for_full_train \
                    = self.auto_config(expected_time_ratio=self.expected_time_ratio,
                                    num_full_epochs=self.num_full_iterations,
                                    full_dataset_size=self.full_dataset_size,
                                    subset_ratio=self.subset_ratio,
                                    non_pure_train_ratio=self.non_pure_train_ratio,
                                    num_hyperparams=len(self.search_space),
                                    reduction_factor=self._reduction_factor,
                                    parallelism=self.num_workers,
                                    min_epochs=self._min_iterations,
                                    max_epochs=self.max_iterations)
            logger.debug(f"AsyncHyperBand.__init__() num_trials : {self.num_trials}, min_iterations : {self._min_iterations}, "
                  f"max_iterations : {self.max_iterations}, subset_ratio {self.subset_ratio}, "
                  f"n_imgs_for_min_train: {self.n_imgs_for_min_train}, n_imgs_for_full_train: {self.n_imgs_for_full_train}, "
                  f"expected_total_images : {self._expected_total_images}, num_brackets: {self._num_brackets}")

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
        # It will be used for generating trials.
        self.bayesopt_space = {}
        for param in self.search_space:
            self.bayesopt_space[param] = (self.search_space[param].lower_space(),
                                          self.search_space[param].upper_space())

        self.optimizer = BayesianOptimization(f=self.obj,
                                              pbounds=self.bayesopt_space,
                                              verbose=self.verbose,
                                              random_state=None)

        self.uf = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0,
                                  kappa_decay=1,
                                  kappa_decay_delay=self.num_init_trials)

        if self.hasCategoricalParam(self.search_space):
            self.optimizer.set_gp_params(alpha=1e-3)

        # All information in self.hpo_status will be stored at hpo_file_path.
        hpo_file_path = hpopt.get_status_path(self.save_path)

        if self.resume is True and os.path.exists(hpo_file_path):
            with open(hpo_file_path, 'rt') as json_file:
                self.hpo_status = json.load(json_file)
                json_file.close()

            if self.hpo_status['search_algorithm'] != 'asha':
                logger.error("Search algorithm is changed. Stop resuming.")
                raise ValueError("Search algorithm is changed.")

            if self.hpo_status['search_space'] != {ss: self.search_space[ss].__dict__ for ss in self.search_space}:
                logger.error("Search space is changed. Stop resuming.")
                raise ValueError("Search space is changed.")

            if self.hpo_status['metric'] != self.metric:
                logger.error("Metric is changed. Stop resuming.")
                raise ValueError("Metric is changed.")

            if self.hpo_status['subset_ratio'] != self.subset_ratio:
                logger.error("subset_ratio is changed. Stop resuming.")
                raise ValueError("subset_ratio is changed.")

            if self.hpo_status['image_resize'] != self.image_resize:
                logger.error("image_resize is changed. Stop resuming.")
                raise ValueError("image_resize is changed.")

            if self.hpo_status['full_dataset_size'] != self.full_dataset_size:
                logger.error("The size of dataset is changed. Stop resuming.")
                raise ValueError("Dataset is changed.")

            if self.hpo_status['reduction_factor'] != self._reduction_factor:
                logger.error("reduction_factor is changed. Stop resuming.")
                raise ValueError("reduction_factor is changed.")
        else:
            self.hpo_status['search_algorithm'] = 'asha'
            self.hpo_status['search_space'] = {ss: self.search_space[ss].__dict__ for ss in self.search_space}
            self.hpo_status['metric'] = self.metric
            self.hpo_status['subset_ratio'] = self.subset_ratio
            self.hpo_status['image_resize'] = self.image_resize
            self.hpo_status['full_dataset_size'] = self.full_dataset_size
            self.hpo_status['reduction_factor'] = self._reduction_factor
            self.hpo_status['config_list'] = []

        # num_gen_config represents the number of assigned configs.
        # If 1 configs are done and 2 configs are running, num_gen_config is 3.
        self.hpo_status['num_gen_config'] = 0

        # Resumes from the previous results.
        for idx, config in enumerate(self.hpo_status['config_list']):
            if config['status'] == hpopt.Status.STOP:
                # This case means that config is done.
                self.hpo_status['num_gen_config'] += 1
            else:
                # This case means that config should be in the ready state.
                config['status'] = hpopt.Status.READY
                trial_file_path = hpopt.get_trial_path(self.save_path, idx)
                if os.path.exists(trial_file_path):
                    # Incomplete informations should be removed before starting.
                    os.remove(trial_file_path)

        # Initialize the brackets
        self.rungs_in_brackets = []
        for s in range(self._num_brackets):
            self.rungs_in_brackets.append(self.get_rungs(self.n_imgs_for_min_train,
                                                         self.n_imgs_for_full_train,
                                                         self._reduction_factor, s))

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
                self.hpo_status['config_list'].append({
                    "trial_id" : idx,
                    "config" : default_hyper_parameter,
                    "status" : hpopt.Status.READY,
                    "score" : None,
                    'bracket': 0,
                })

        num_ready_configs = len(self.hpo_status['config_list'])

        for i in range(len(n0_in_brackets)):
            n0_in_brackets[i] = int(brackets_ratio[i] * self.num_trials)

        n0_in_brackets[0] += (self.num_trials - sum(n0_in_brackets))

        for trial in self.hpo_status['config_list']:
            n0_in_brackets[trial['bracket']] -= 1

        # Generate trial configs up to self.num_trials
        for i in range(num_ready_configs, self.num_trials):
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
                config = self.get_real_config(self.optimizer.suggest(self.uf))
                duplicated = self.check_duplicated_config(config)
                retry_count += 1

                if retry_count > 100:
                    return None

            self.hpo_status['config_list'].append(
                    {'trial_id': i,
                     'config': config,
                     'status': hpopt.Status.READY,
                     'score': None,
                     'bracket': bracket_id}
                )

        # Sync the current status to the file system.
        self.save_results()

    # Get the list of iteration numbers for rungs in the bracket
    @staticmethod
    def get_rungs(min_t: int, max_t: int, rf: int, s: int):
        MAX_RUNGS = int(log(max_t - min_t + 1) / log(rf) - s + 1)
        rungs = [(min_t - 1) + rf**(k + s) for k in reversed(range(MAX_RUNGS))]

        return rungs

    def get_num_trials_in_brackets(self, reduction_factor: int, num_brackets: int):
        brackets = []

        for s in reversed(range(num_brackets)):
            n = int(ceil(int(num_brackets/(s+1))*reduction_factor**s))
            brackets.append(n)

        return brackets

    def save_results(self):
        hpo_file_path = hpopt.get_status_path(self.save_path)
        oldmask = os.umask(0o077)
        with open(hpo_file_path, 'wt') as json_file:
            json.dump(self.hpo_status, json_file, indent=4)
            json_file.close()
        os.umask(oldmask)

    def get_next_sample(self):
        # Gather all results from workers
        self.update_scores()

        # Check if unassigned configs
        if self.hpo_status['num_gen_config'] >= self.num_trials:
            return None

        # Check total number of trained images
        if self._expected_total_images > 0:
            num_trained_images = self.get_num_trained_images()
            logger.debug(f"expected total images = {self._expected_total_images} "
                  f"num_images = {num_trained_images}")
            if num_trained_images >= self._expected_total_images:
                return None

        # Choose a config
        next_config = None
        trial_id = None
        for idx, config in enumerate(self.hpo_status['config_list']):
            if config['status'] is hpopt.Status.READY:
                config['status'] = hpopt.Status.RUNNING
                next_config = config
                trial_id = idx
                break

        if next_config is None:
            return None

        self.hpo_status['num_gen_config'] += 1
        self.save_results()

        new_config = {}
        new_config['params'] = next_config['config']
        new_config['iterations'] = self.max_iterations
        new_config['trial_id'] = trial_id
        new_config['bracket'] = next_config['bracket']
        new_config['rungs'] = self.rungs_in_brackets[next_config['bracket']]
        new_config['file_path'] = hpopt.get_trial_path(self.save_path, new_config['trial_id'])
        new_config['subset_ratio'] = self.subset_ratio
        new_config['resize_width'] = self.image_resize[0]
        new_config['resize_height'] = self.image_resize[1]
        # added to calculate number of iterations
        new_config['dataset_size'] = self.full_dataset_size
        new_config['mode'] = self.mode
        new_config['iteration_limit'] = self.n_imgs_for_full_train
        new_config['batch_size_param_name'] = self.batch_size_name

        return new_config

    def update_scores(self):
        for trial_id, config_item in enumerate(self.hpo_status['config_list']):
            if config_item['status'] == hpopt.Status.RUNNING:
                current_status = hpopt.get_current_status(self.save_path, trial_id)

                if current_status == hpopt.Status.STOP:
                    score = hpopt.get_best_score(self.save_path, trial_id, self.mode)

                    if score is not None:
                        config_item['score'] = score
                        config_item['status'] = hpopt.Status.STOP
                        real_config = config_item['config']
                        logger.info(f'#{trial_id} | {real_config} | {score}')
                elif current_status == hpopt.Status.CUDAOOM:
                    config_item['status'] = hpopt.Status.READY
                    self.hpo_status['num_gen_config'] -= 1

                    trial_file_path = hpopt.get_trial_path(self.save_path, trial_id)
                    if os.path.exists(trial_file_path):
                        # Failed information should be removed before starting.
                        os.remove(trial_file_path)

                    # shrink search space of 'bs'
                    self.shrink_bs_search_space(config_item)

        self.save_results()

        # If the current schedule is updatable, adjust it regarding to the latest information
        if self._updatable_schedule is True:
            logger.info("updating schedule")

            _, _, _, new_expected_total_images, _, _, _, _  = self.auto_config(
                expected_time_ratio=self.expected_time_ratio,
                num_full_epochs=self.num_full_iterations,
                full_dataset_size=self.full_dataset_size,
                subset_ratio=self.subset_ratio,
                non_pure_train_ratio=self.non_pure_train_ratio,
                num_hyperparams=len(self.search_space),
                reduction_factor=self._reduction_factor,
                parallelism=self.num_workers,
                min_epochs=self._min_iterations,
                max_epochs=self.max_iterations)
            logger.debug(f"updated expected total images from {self._expected_total_images} "
                  f"to {new_expected_total_images}")
            self._expected_total_images = new_expected_total_images

    # Lower the upper bound of batch size
    def shrink_bs_search_space(self, not_allowed_config):
        # Stop if batch_size_name is not specified.
        if self.batch_size_name is None:
            return

        # Check if batch-size is in the list of tuning params.
        if self.batch_size_name not in self.search_space:
            return

        # Check if the search space type for batch size if qloguniform or quniform.
        if self.search_space[self.batch_size_name].type not in ['qloguniform', 'quniform']:
            return

        not_allowed_bs = not_allowed_config['config'][self.batch_size_name]
        new_upper_bound = not_allowed_bs - self.search_space[self.batch_size_name].range[2]

        # if new_upper_bound is greater than the current upper bound, update only this trial.
        if self.search_space[self.batch_size_name].range[1] <= new_upper_bound:
            not_allowed_config['config'] = self.get_real_config(self.optimizer.suggest(self.uf))
            return

        self.search_space[self.batch_size_name].range[1] = new_upper_bound

        # if the new upper bound is less than the current lower bound,
        # update the lower bound to be half of the new upper bound.
        if self.search_space[self.batch_size_name].range[0] > self.search_space[self.batch_size_name].range[1]:
            new_lower_bound = self.search_space[self.batch_size_name].range[1] // 2
            self.search_space[self.batch_size_name].range[0] = max(new_lower_bound, 2)

            if self.search_space[self.batch_size_name].range[0] > self.search_space[self.batch_size_name].range[1]:
                raise ValueError('This model cannot be trained even with batch size of 2.')

        # Reset search space
        self.bayesopt_space = {}
        for param in self.search_space:
            self.bayesopt_space[param] = (self.search_space[param].lower_space(),
                                          self.search_space[param].upper_space())

        self.optimizer = BayesianOptimization(f=self.obj,
                                              pbounds=self.bayesopt_space,
                                              verbose=self.verbose,
                                              random_state=None)

        if self.hasCategoricalParam(self.search_space):
            self.optimizer.set_gp_params(alpha=1e-3)

        for idx, config in enumerate(self.hpo_status['config_list']):
            if config['status'] is hpopt.Status.READY:
                config['config'] = self.get_real_config(self.optimizer.suggest(self.uf))

    def get_num_trained_images(self):
        num_trained_images = 0

        for trial_id, config_item in enumerate(self.hpo_status['config_list']):
            trial_file_path = hpopt.get_trial_path(self.save_path, trial_id)
            trial_results = hpopt.load_json(trial_file_path)
            if trial_results is not None:
                images = trial_results.get('images', None)
                if images is not None:
                    num_trained_images += images[-1]

        return num_trained_images

    def auto_config(self,
                    expected_time_ratio: Union[int, float],
                    num_full_epochs: int,
                    full_dataset_size: Optional[int],
                    subset_ratio: Optional[Union[float, int]],
                    non_pure_train_ratio: float,
                    num_hyperparams: int,
                    reduction_factor: int,
                    parallelism: int,
                    min_epochs: Optional[int],
                    max_epochs: Optional[int]):
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
        if max_epochs is None:
            # max_iterations = num_full_epochs
            max_epochs = num_full_epochs

        if min_epochs is None:
            min_epochs = 1

        if min_epochs > max_epochs:
            logger.info("min_epochs should be less than or equal to max_epochs.")
            min_epochs = max_epochs

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
        logger.debug(f"n_imgs_for_full_train = {n_imgs_for_full_train}, "
              f"n_imgs_for_min_train = {n_imgs_for_min_train}")

        # num_trials = 0
        # num_brackets = 1
        # current_time_ratio = expected_time_ratio * 0.5

        # Update num_full_images from previous trials
        max_num_images_in_trials = 0

        if self.hpo_status.get('config_list', None) is not None:
            for trial_id, config_item in enumerate(self.hpo_status['config_list']):
                trial_file_path = hpopt.get_trial_path(self.save_path, trial_id)
                trial_results = hpopt.load_json(trial_file_path)
                if trial_results is not None:
                    n_images = trial_results.get('images', None)
                    if n_images is not None:
                        if n_images[-1] > max_num_images_in_trials:
                            max_num_images_in_trials = n_images[-1]

            logger.debug(f"(before) max_num_images_in_trials: {max_num_images_in_trials}")

            if max_num_images_in_trials > 0:
                rungs = self.get_rungs(
                    max_t=n_imgs_for_full_train,
                    min_t=n_imgs_for_min_train,
                    rf=reduction_factor,
                    s=0
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
        num_trials, num_total_images = self._calc_total_budget(num_brackets,
                                                               expected_time_ratio,
                                                               reduction_factor,
                                                               n_imgs_for_min_train,
                                                               n_imgs_for_full_train,
                                                               subset_ratio,
                                                               non_pure_train_ratio,
                                                               parallelism)

        logger.debug(f"auto_config() results: num_trials ={num_trials}, "
              f"max_epochs = {max_epochs}, min_epochs = {min_epochs}, "
              f"n_imgs_for_full_train {n_imgs_for_full_train}, "
              f"n_imgs_for_min_train = {n_imgs_for_min_train}, "
              f"num_total_images = {num_total_images}, "
              f"num_brackets = {num_brackets}, subset_ratio = {subset_ratio}")

        return (
            num_trials,
            max_epochs,
            subset_ratio,
            num_total_images,
            num_brackets,
            min_epochs,
            n_imgs_for_min_train,
            n_imgs_for_full_train
        )

    def _calc_total_budget(self,
                           num_brackets,
                           time_ratio,
                           reduction_factor,
                           resource_min,
                           resource_max,
                           subset_ratio,
                           non_trainset_ratio,
                           parallelism):
        num_trials = 0
        current_time_ratio = time_ratio * 0.5
        while current_time_ratio < time_ratio:
            num_trials = num_trials + 1
            num_total_images = self.get_total_n_images(num_trials, reduction_factor,
                                                   num_brackets, resource_min, resource_max)

            current_time_ratio = num_total_images / resource_max / parallelism * \
                ((1 - non_trainset_ratio) * subset_ratio + non_trainset_ratio)
        return num_trials, num_total_images

    def get_total_n_images(self,
                           num_trials: int,
                           reduction_factor: int,
                           num_brackets: int,
                           n_imgs_min_train: int,
                           n_imgs_full_train:int):
        num_total_images = 0

        num_trials_in_brackets = self.get_num_trials_in_brackets(reduction_factor, num_brackets)
        # logger.debug(f"num_trials_in_brackets = {num_trials_in_brackets}")

        brackets_total = sum(num_trials_in_brackets)
        brackets_ratio = [float(b / brackets_total) for b in num_trials_in_brackets]

        for i in range(len(num_trials_in_brackets)):
            num_trials_in_brackets[i] = int(brackets_ratio[i] * num_trials)

        num_trials_in_brackets[0] += (num_trials - sum(num_trials_in_brackets))

        for s, num_trials in enumerate(num_trials_in_brackets):
            rungs = self.get_rungs(n_imgs_min_train, n_imgs_full_train, reduction_factor, s)
            remained_trials = num_trials
            for rung in reversed(rungs):
                num_stop_trials = remained_trials - (remained_trials // reduction_factor)
                num_total_images += (num_stop_trials * rung)
                remained_trials = remained_trials // reduction_factor
            num_total_images += (remained_trials * n_imgs_full_train)

        return num_total_images

    def get_progress(self):
        # num images based progress
        image_progress = min(self.get_num_trained_images() / self._expected_total_images, 0.99)
        # trial based progress
        finished_trials = sum([val['status'] == hpopt.Status.STOP
                    for val in self.hpo_status['config_list']])
        trial_progress = finished_trials / self.num_trials

        # return min(0.99, max(epoch_progress, trial_progress))
        logger.debug(f"get_progress() iter = {image_progress}/{self._expected_total_images}"
              f", trial {trial_progress}/{self.num_trials}")

        return min(0.99, max(image_progress, trial_progress))
