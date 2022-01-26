# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
import logging
from typing import Optional, Union

from bayes_opt import BayesianOptimization, UtilityFunction

from hpopt.hpopt_abstract import HpOpt
import hpopt

logger = logging.getLogger(__name__)


class BayesOpt(HpOpt):
    """
    This implements the Bayesian optimization. Bayesian optimization is
    sequantial optimization method. Previous results affects which hyper parameter
    to test using gaussian process which utilizes relationship between hyper parameters.

    Args:
        early_stop (bool): early stop flag.
        kappa (float or int): Kappa vlaue for ucb used in bayesian optimization.
        kappa_decay (float or int): Multiply kappa by kappa_decay every trials.
        kappa_decay_delay (int): From first trials to kappa_decay_delay trials,
                                 kappa isn't multiplied to kappa_decay.
    """
    def __init__(self,
                 early_stop: Optional[bool] = None,
                 kappa: Union[float, int] = 2.576,
                 kappa_decay: int = 1,
                 kappa_decay_delay: int = 0,
                 **kwargs):
        super(BayesOpt, self).__init__(**kwargs)

        self.updatable_schedule = False
        self.early_stop = early_stop

        # HPO auto configurator
        if self.num_trials is None or self.max_iterations is None or self.subset_ratio is None:
            self.updatable_schedule = True
            self.num_trials, self.max_iterations, self.subset_ratio = \
                self.auto_config(self.expected_time_ratio,
                                 self.num_full_iterations,
                                 self.full_dataset_size,
                                 self.non_pure_train_ratio,
                                 len(self.search_space),
                                 self.num_trials,
                                 self.max_iterations,
                                 self.subset_ratio,
                                 self.num_workers)
            logger.debug(f"auto-config: num_trials {self.num_trials} "
                         f"max_iterations {self.max_iterations} subset_ratio {self.subset_ratio}")

        if self.num_init_trials > self.num_trials:
            self.num_init_trials = self.num_trials

        self.bayesopt_space = {}
        for param in self.search_space:
            self.bayesopt_space[param] = (self.search_space[param].lower_space(),
                                          self.search_space[param].upper_space())

        self.optimizer = BayesianOptimization(f=self.obj,
                                              pbounds=self.bayesopt_space,
                                              verbose=self.verbose,
                                              random_state=None)

        self.uf = UtilityFunction(kind="ucb", xi=0.0,
                                  kappa=kappa,
                                  kappa_decay=kappa_decay,
                                  kappa_decay_delay=kappa_decay_delay)

        if self.hasCategoricalParam(self.search_space):
            self.optimizer.set_gp_params(alpha=1e-3)

        hpo_file_path = hpopt.get_status_path(self.save_path)

        if self.resume is True and os.path.exists(hpo_file_path):
            with open(hpo_file_path, 'rt') as json_file:
                self.hpo_status = json.load(json_file)
                json_file.close()

            if self.hpo_status['search_algorithm'] != 'smbo':
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

            if self.hpo_status['early_stop'] != self.early_stop:
                logger.error("early_stop is changed. Stop resuming.")
                raise ValueError("early_stop is changed.")

            if self.hpo_status['max_iterations'] != self.max_iterations:
                logger.error("max_iterations is changed. Stop resuming.")
                raise ValueError("max_iterations is changed.")

            if self.hpo_status['full_dataset_size'] != self.full_dataset_size:
                logger.error("The size of dataset is changed. Stop resuming.")
                raise ValueError("Dataset is changed.")
        else:
            self.hpo_status['search_algorithm'] = 'smbo'
            self.hpo_status['search_space'] = {ss: self.search_space[ss].__dict__ for ss in self.search_space}
            self.hpo_status['metric'] = self.metric
            self.hpo_status['subset_ratio'] = self.subset_ratio
            self.hpo_status['image_resize'] = self.image_resize
            self.hpo_status['early_stop'] = self.early_stop
            self.hpo_status['max_iterations'] = self.max_iterations
            self.hpo_status['full_dataset_size'] = self.full_dataset_size
            self.hpo_status['config_list'] = []

        self.hpo_status['num_gen_config'] = 0

        for idx, config in enumerate(self.hpo_status['config_list'], start=0):
            if config['status'] == hpopt.Status.STOP:
                self.hpo_status['num_gen_config'] += 1
                self.optimizer.register(params=self.get_space_config(config['config']), target=config['score'])
                self.uf.update_params()
            else:
                config['status'] = hpopt.Status.READY
                trial_file_path = hpopt.get_trial_path(self.save_path, idx)
                if os.path.exists(trial_file_path):
                    os.remove(trial_file_path)

        num_ready_configs = len(self.hpo_status['config_list'])

        for i in range(num_ready_configs, self.num_init_trials):
            # Generate a new config
            duplicated = True
            retry_count = 0

            # Check if the new config is duplicated
            while duplicated is True:
                config = self.get_real_config(self.optimizer.suggest(self.uf))
                duplicated = self.check_duplicated_config(config)
                retry_count += 1

                if retry_count > 10:
                    break

            if duplicated is True:
                continue

            self.hpo_status['config_list'].append(
                    {'trial_id': i,
                     'config': config,
                     'status': hpopt.Status.READY,
                     'score': None}
                )

        self.save_results()

    def save_results(self):
        hpo_file_path = hpopt.get_status_path(self.save_path)
        oldmask = os.umask(0o077)
        with open(hpo_file_path, 'wt') as json_file:
            json.dump(self.hpo_status, json_file, indent=4)
            json_file.close()
        os.umask(oldmask)

    def update_scores(self):
        for trial_id, config_item in enumerate(self.hpo_status['config_list'], start=0):
            if config_item['status'] == hpopt.Status.RUNNING:
                current_status = hpopt.get_current_status(self.save_path, trial_id)

                if current_status == hpopt.Status.STOP:
                    score = hpopt.get_best_score(self.save_path, trial_id, self.mode)

                    if score is not None:
                        config_item['score'] = score
                        if self.mode == 'min':
                            self.optimizer.register(params=self.get_space_config(config_item['config']), target=-score)
                        else:
                            self.optimizer.register(params=self.get_space_config(config_item['config']), target=score)
                        self.uf.update_params()
                        config_item['status'] = hpopt.Status.STOP
                        real_config = config_item['config']
                        logging.info(f'#{trial_id} | {real_config} | {score}')
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

    # Lower the upper bound of batch size
    def shrink_bs_search_space(self, not_allowed_config):
        # Stop if batch_size_name is not specified.
        if self.batch_size_name is None:
            return

        # Check if batch-size is in the list of tuning params.
        if self.batch_size_name not in self.search_space:
            return

        # Check if the search space type for batch size if qloguniform.
        if self.search_space[self.batch_size_name].type not in ['qloguniform', 'quniform']:
            return

        not_allowed_bs = not_allowed_config['config'][self.batch_size_name]
        new_upper_bound = not_allowed_bs - self.search_space[self.batch_size_name].range[2]

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

        del_list = []

        for idx, config in enumerate(self.hpo_status['config_list']):
            if config['status'] is hpopt.Status.READY:
                # Generate a new config
                duplicated = True
                retry_count = 0
                new_config = None

                # Check if the new config is duplicated
                while duplicated is True:
                    new_config = self.get_real_config(self.optimizer.suggest(self.uf))
                    duplicated = self.check_duplicated_config(new_config)
                    retry_count += 1

                    if retry_count > 10:
                        del_list.append(idx)
                        break

                config['config'] = new_config

        # remove duplicated configs
        for idx in reversed(del_list):
            del self.hpo_status['config_list'][idx]

        for config in self.hpo_status['config_list']:
            if config['status'] == hpopt.Status.STOP:
                self.optimizer.register(params=self.get_space_config(config['config']), target=config['score'])

        self.save_results()


    def auto_config(self,
                    expected_time_ratio: Union[int, float],
                    num_full_iterations: int,
                    full_dataset_size: Optional[int],
                    non_pure_train_ratio: float,
                    num_hyperparams: int,
                    num_trials: Optional[int],
                    max_iterations: Optional[int],
                    subset_ratio: Optional[Union[float, int]],
                    parallelism: int = 1):
        # All arguments should be specified.
        if expected_time_ratio is None:
            raise ValueError("expected_time_ratio should be specified.")

        if num_full_iterations is None:
            raise ValueError("num_full_iterations should be specified.")

        if non_pure_train_ratio is None:
            raise ValueError("non_pure_train_ratio should be specified.")

        if num_hyperparams is None:
            raise ValueError("num_hyperparams should be specified.")

        # 1. Config target parameters as small as possible
        if max_iterations is None:
            max_iterations = min(2, num_full_iterations)

        if subset_ratio is None:
            subset_ratio = 0.2

        # Make subset_ratio to be between 0.2 and 1.0
        subset_ratio = min(1.0, subset_ratio)
        subset_ratio = max(0.2, subset_ratio)

        if full_dataset_size == 0:
            logger.warning("Sub-dataset isn't used because full_dataset_size value is 0.")
            subset_ratio = 1.0
        elif (full_dataset_size * subset_ratio) < self.min_subset_size:
            if full_dataset_size > self.min_subset_size:
                subset_ratio = float(self.min_subset_size) / full_dataset_size
            else:
                subset_ratio = 1.0

        if num_trials is None:
            num_trials = 10

        current_time_ratio = num_trials * max_iterations / num_full_iterations * \
            ((1 - non_pure_train_ratio) * subset_ratio + non_pure_train_ratio)

        # If current_time_ratio is less than expected_time_ratio,
        # return the minimum config values
        if current_time_ratio >= expected_time_ratio:
            return num_trials, max_iterations, subset_ratio

        # 2. Update the target parameters iteratively
        iter_ratio = max_iterations / num_full_iterations
        max_trials = 20 * parallelism
        max_trials *= num_hyperparams
        trial_ratio = num_trials / max_trials

        while current_time_ratio < expected_time_ratio:
            if subset_ratio < 1.0 and subset_ratio <= (5 * iter_ratio) and subset_ratio < trial_ratio:
                subset_ratio = subset_ratio * 1.05
                subset_ratio = min(1.0, subset_ratio)
            elif iter_ratio < 1.0 and (5 * iter_ratio) <= subset_ratio and (5 * iter_ratio) < trial_ratio:
                max_iterations += 1
                max_iterations = min(max_iterations, num_full_iterations)
            else:
                num_trials += 1

            iter_ratio = max_iterations / num_full_iterations
            trial_ratio = num_trials / max_trials

            current_time_ratio = num_trials * iter_ratio * \
                ((1 - non_pure_train_ratio) * subset_ratio + non_pure_train_ratio)

            if subset_ratio > 0.99 and iter_ratio > 0.99 and trial_ratio > 0.99:
                subset_ratio = 1.0
                break

        return num_trials, max_iterations, subset_ratio

    #
    # Retrieve the next config to evaluate
    #
    def get_next_sample(self):
        if self.hpo_status['num_gen_config'] >= self.num_trials:
            return None

        self.update_scores()

        config = None

        if self.hpo_status['num_gen_config'] < len(self.hpo_status['config_list']):
            # Choose a config that has not yet been evaluated from the config_list
            num_done_config = 0
            for gen_config in self.hpo_status['config_list']:
                if gen_config['status'] == hpopt.Status.STOP:
                    num_done_config += 1

            for idx, gen_config in enumerate(self.hpo_status['config_list'], start=0):
                if gen_config['status'] == hpopt.Status.READY:
                    config = gen_config['config']
                    gen_config['status'] = hpopt.Status.RUNNING
                    trial_id = idx
                    break
        else:
            # Generate a new config
            duplicated = True
            retry_count = 0

            # Check if the new config is duplicated
            while duplicated is True:
                config = self.get_real_config(self.optimizer.suggest(self.uf))
                duplicated = self.check_duplicated_config(config)
                retry_count += 1

                if retry_count > 10:
                    return None

            trial_id = self.hpo_status['num_gen_config']
            self.hpo_status['config_list'].append(
                    {'trial_id': trial_id,
                     'config': config,
                     'status': hpopt.Status.RUNNING,
                     'score': None}
                )

        if config is None:
            return None

        self.hpo_status['num_gen_config'] += 1

        new_config = {}
        new_config['params'] = config
        new_config['iterations'] = self.max_iterations
        new_config['trial_id'] = trial_id
        new_config['file_path'] = hpopt.get_trial_path(self.save_path, new_config['trial_id'])

        if os.path.exists(new_config['file_path']):
            os.remove(new_config['file_path'])

        if self.hpo_status['num_gen_config'] <= self.num_init_trials:
            new_config['early_stop'] = None
        else:
            new_config['early_stop'] = self.early_stop

        new_config['subset_ratio'] = self.subset_ratio
        new_config['resize_width'] = self.image_resize[0]
        new_config['resize_height'] = self.image_resize[1]
        new_config['mode'] = self.mode

        self.save_results()

        return new_config

    def get_next_samples(self, num_expected_samples=0):
        new_configs = []

        while self.hpo_status['num_gen_config'] < self.num_init_trials:
            new_configs.append(self.get_next_sample())

        if len(new_configs) == 0:
            new_config = self.get_next_sample()

            if new_config is not None:
                new_configs.append(new_config)

        return new_configs

    def get_progress(self):
        finished_trials = sum([val['status'] == hpopt.Status.STOP
                               for val in self.hpo_status['config_list']])
        progress = finished_trials / self.num_trials

        final_trial_path = hpopt.get_trial_path(self.save_path, finished_trials)
        if os.path.exists(final_trial_path):
            with open(final_trial_path, 'r') as f:
                progress += (len(json.load(f)['scores'])
                            / (self.num_trials * self.max_iterations))
        
        return min(progress, 0.99)
