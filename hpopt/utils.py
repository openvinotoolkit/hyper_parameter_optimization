# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import json
import os
import time
from enum import IntEnum
from statistics import median
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import transforms

import hpopt
from hpopt.dummy import DummyOpt
from hpopt.hyperband import AsyncHyperBand
from hpopt.logger import get_logger
from hpopt.smbo import Smbo

logger = get_logger()


def createDummyOpt(
    search_alg=None, save_path=None, search_space=None, resume=False, **kwargs
):
    if save_path is None:
        return None

    return DummyOpt(save_path=save_path, search_space=search_space, resume=resume)


def create(
    full_dataset_size: int,
    num_full_iterations: int,
    search_space: Dict[str, Dict[str, Any]],
    save_path: str = "hpo",
    search_alg: str = "bayes_opt",
    early_stop: Optional[bool] = None,
    mode: str = "max",
    num_init_trials: int = 5,
    num_trials: Optional[int] = None,
    max_iterations: Optional[int] = None,
    min_iterations: Optional[int] = None,
    reduction_factor: int = 2,
    num_brackets: Optional[int] = None,
    subset_ratio: Optional[Union[float, int]] = None,
    batch_size_name: str = None,
    image_resize: List[int] = [0, 0],
    metric: str = "mAP",
    resume: bool = False,
    expected_time_ratio: Union[int, float] = 4,
    non_pure_train_ratio: float = 0.2,
    num_workers: int = 1,
    kappa: Union[float, int] = 2.576,
    kappa_decay: Union[float, int] = 1,
    kappa_decay_delay: int = 0,
    default_hyper_parameters: Optional[Union[List[Dict], Dict]] = None,
    use_epoch: Optional[bool] = False,
):
    """
    Create a new hpopt instance.

    Args:
        full_dataset_size (int): train dataset size.
        num_full_iterations (int): epoch for traninig after HPO.
        save_path (str): path where result of HPO is saved.
        search_alg (str): bayes_opt or asha.
                          search algorithm to use for optimizing hyper parmaeters.
                          Hpopt support SMBO and ASHA(HyperBand).
        search_space (list): hyper parameter search space to find.
        early_stop (bool): early stop flag.
        mode (str): One of {min, max}. Determines whether objective is
                    minimizing or maximizing the metric attribute.
        num_init_trials (int): Only for SMBO. How many trials to use to init SMBO.
        num_trials (int): How many training to conduct for HPO.
        max_iterations (int): Max training iterations for each trial.
        min_iterations (int): Only for ASHA. Only stop trials at least this old in time.
        reduction_factor (int): Only for ASHA. Used to set halving rate and amount.
                                This is simply a unit-less scalar.
        num_brackets (int): Only for ASHA. Bracket number of ASHA(Hyperband).
                            Each bracket has a different halving rate,
                            specified by the reduction factor.
        subset_ratio (float or int): ratio to how many train dataset to use for each trial.
                                     The lower value is, the faster the speed is.
                                     But If it's too low, HPO can be unstable.
                                     Whatever value is, minimum dataset size is 500.
        image_resize (list): Width and height of image used to resize for decreasing HPO time.
                             First value of list is width and second value is height.
        resume (bool): resume flag decide to use previous HPO results.
                       If HPO completed, you can just use optimized hyper parameters.
                       If HPO stopped in middle, you can resume in middle.
        expected_time_ratio (int or float): Time to use for HPO.
                                            If HPO is configured automatically,
                                            HPO use time about exepected_time_ratio *
                                            train time after HPO times.
        non_pure_train_ratio (float): ratio of validation time to (train time + validation time)
        num_workers (int): How many trains are executed in parallel.
        kappa (float or int): Only for SMBO. Kappa vlaue for ucb used in bayesian optimization.
        kappa_decay (float or int): Only for SMBO. Multiply kappa by kappa_decay every trials.
        kappa_decay_delay (int): Only for SMBO. From first trials to kappa_decay_delay trials,
                                 kappa isn't multiplied to kappa_decay.
        default_hyper_parameters (List[Dict] or Dict): default hyper-parameters.
        use_epoch (bool): use epoch unit instead of epoch.
    """
    logger.info(f"creating hpopt instance with {search_alg} algo")
    os.makedirs(save_path, exist_ok=True)
    if resume is False:
        status_path = get_status_path(save_path)
        if os.path.exists(status_path):
            os.remove(status_path)
        clear_trial_files(save_path)

    if search_alg == "bayes_opt":
        return Smbo(
            save_path=save_path,
            search_space=search_space,
            early_stop=early_stop,
            mode=mode,
            num_init_trials=num_init_trials,
            num_trials=num_trials,
            max_iterations=max_iterations,
            subset_ratio=subset_ratio,
            batch_size_name=batch_size_name,
            image_resize=image_resize,
            metric=metric,
            resume=resume,
            expected_time_ratio=expected_time_ratio,
            num_full_iterations=num_full_iterations,
            full_dataset_size=full_dataset_size,
            non_pure_train_ratio=non_pure_train_ratio,
            num_workers=num_workers,
            kappa=kappa,
            kappa_decay=kappa_decay,
            kappa_decay_delay=kappa_decay_delay,
            default_hyper_parameters=default_hyper_parameters,
        )
    elif search_alg == "asha":
        return AsyncHyperBand(
            save_path=save_path,
            search_space=search_space,
            mode=mode,
            num_trials=num_trials,
            max_iterations=max_iterations,
            min_iterations=min_iterations,
            reduction_factor=reduction_factor,
            num_brackets=num_brackets,
            subset_ratio=subset_ratio,
            batch_size_name=batch_size_name,
            image_resize=image_resize,
            metric=metric,
            resume=resume,
            expected_time_ratio=expected_time_ratio,
            num_full_iterations=num_full_iterations,
            full_dataset_size=full_dataset_size,
            non_pure_train_ratio=non_pure_train_ratio,
            num_workers=num_workers,
            default_hyper_parameters=default_hyper_parameters,
            use_epoch=use_epoch,
        )
    else:
        raise ValueError(f"Not supported search algorithm: {search_alg}")


def get_status_path(save_path: str):
    """
    Return path of HPO status file

    Args:
        save_path (str): path where result of HPO is saved.
    """
    return os.path.join(save_path, "hpopt_status.json")


def get_trial_path(save_path: str, trial_id: int):
    """
    Return path of HPO trial file

    Args:
        save_path (str): path where result of HPO is saved.
        tiral_id (int): order of HPO trial
    """
    return os.path.join(save_path, f"hpopt_trial_{trial_id}.json")


def clear_trial_files(save_path: str):
    """
    clear HPO trial files

    Args:
        save_path (str): path where result of HPO is saved.
    """
    trial_file_list = glob.glob(os.path.join(save_path, "hpopt_trial_*.json"))

    for trial_file_path in trial_file_list:
        try:
            os.remove(trial_file_path)
        except OSError:
            logger.error(f"Error while deleting file : {trial_file_path}")
            return None


# hpopt.Status.COMPLETERESULT - Previous HPO task is completely finished.
#                              It means that the best hyper-parameters are already found.
#
# hpopt.Status.PARTIALRESULT - Previous HPO task is started but not finished yet.
#
# hpopt.Status.NORESULT - Previous HPO task is not found.
#
def get_previous_status(save_path: str):
    """
    Check if there is previous results.

    Args:
        save_path (str): path where result of HPO is saved.
    """
    status_file_path = get_status_path(save_path)

    if os.path.exists(status_file_path):
        best_config_id = None
        with open(status_file_path, "rt") as json_file:
            hpo_status = json.load(json_file)
            best_config_id = hpo_status.get("best_config_id", None)

        if best_config_id is not None:
            return hpopt.Status.COMPLETERESULT
        else:
            return hpopt.Status.PARTIALRESULT
    else:
        return hpopt.Status.NORESULT


def get_current_status(save_path: str, trial_id: int):
    """
    Return status of HPO trial.

    Args:
        save_path (str): path where result of HPO is saved.
        tiral_id (int): order of HPO trial.
    """
    trial_file_path = get_trial_path(save_path, trial_id)
    trial_results = load_json(trial_file_path)

    if trial_results is not None:
        return trial_results["status"]
    else:
        return hpopt.Status.UNKNOWN


def get_best_score(save_path: str, trial_id: int, mode: str):
    """
    Return status of HPO trial.

    Args:
        save_path (str): path where result of HPO is saved.
        tiral_id (int): order of HPO trial.
        mode (str): max or min. Decide whether to find max value or min value.
    """
    trial_file_path = get_trial_path(save_path, trial_id)
    trial_results = load_json(trial_file_path)

    if trial_results is not None:
        if trial_results["status"] == Status.STOP:
            if mode == "min":
                return min(trial_results["scores"])
            else:
                return max(trial_results["scores"])

    return None


def get_best_score_with_num_imgs(save_path: str, trial_id: int, mode: str):
    """
    get the best score of the trial.

    Args:
        save_path (str): path where result of HPO is saved.
        tiral_id (int): order of HPO trial.
        mode (str): max or min. Decide whether to find max value or min value.
    """
    trial_file_path = get_trial_path(save_path, trial_id)
    trial_results = load_json(trial_file_path)

    best_score = None
    num_images = -1
    if trial_results is not None:
        if trial_results["status"] == Status.STOP:
            if mode == "min":
                best_score = min(trial_results["scores"])
            else:
                best_score = max(trial_results["scores"])
            num_images = trial_results["images"][-1]

    return best_score, num_images


def finalize_trial(config: Dict[str, Any]):
    """
    Handles the status of trials that have terminated by unexpected causes

    Args:
        config (dict): HPO configuration for a trial.
                       This include train confiuration(e.g. hyper parameter, epoch, etc.)
                       and tiral information.
    """
    trial_results = load_json(config["file_path"])

    if trial_results is not None:
        if trial_results["status"] != Status.STOP:
            trial_results["status"] = Status.STOP

            # Check if the trial is terminated by the EarlyStoppingHook
            scores_list = trial_results["scores"]
            if len(scores_list) > 5:
                earlyStopped = True
                for i in range(len(scores_list) - 1, len(scores_list) - 6, -1):
                    if scores_list[i] > scores_list[i - 1]:
                        earlyStopped = False
                        break

                if earlyStopped:
                    logger.debug("This trial is earlystopped")
                    trial_results["early_stopped"] = 1

            oldmask = os.umask(0o077)
            with open(config["file_path"], "wt") as json_file:
                json.dump(trial_results, json_file, indent=4)
                json_file.flush()
            os.umask(oldmask)


def get_median_score(save_path: str, trial_id: int, curr_iteration: int):
    """
    Return median score at curr_iteration epoch from previous trials

    Args:
        save_path (str): path where result of HPO is saved.
        tiral_id (int): order of HPO trial
        curr_iteration (int): which epoch to get score at from previous trials
    """
    median_score_list = []

    files = os.listdir(save_path)
    for f in files:
        if f.startswith("hpopt_trial") is False:
            continue

        if f != f"hpopt_trial_{trial_id}.json":
            trial_file_path = os.path.join(save_path, f)
            try:
                with open(trial_file_path, "rt") as json_file:
                    trial_results = json.load(json_file)

                    if len(trial_results["median"]) >= curr_iteration:
                        median_score_list.append(
                            trial_results["median"][curr_iteration - 1]
                        )
            except FileNotFoundError:
                continue

    if len(median_score_list) == 0:
        return None
    else:
        return median(median_score_list)


def get_cutoff_score(save_path: str, target_rung: int, _rung_list, mode: str):
    """
    Calculate a cutoff score for a specified rung of ASHA

    Args:
        save_path (str): path where result of HPO is saved.
        target_rung (int): current rung number
        _rung_list (list): list of rungs in the current bracket
        mode (str): max or min. Decide whether to find max value or min value.
    """
    logger.debug(f"get_cutoff_score({target_rung}, {_rung_list}, {mode})")
    status_file_path = get_status_path(save_path)
    if not os.path.exists(status_file_path):
        logger.warning(f"not existed status json file {status_file_path}")
        return None

    hpo_status = load_json(status_file_path)

    if hpo_status is None:
        logger.warning(f"failed to load json file {status_file_path}")
        return None

    # Gather all scores with iter number
    hpo_trial_results_scores = []
    hpo_trial_results_imgs_num = []

    for idx, config in enumerate(hpo_status["config_list"]):
        if config["status"] in [hpopt.Status.RUNNING, hpopt.Status.STOP]:
            trial_file_path = hpopt.get_trial_path(save_path, idx)
            trial_results = load_json(trial_file_path)

            if trial_results is not None:
                hpo_trial_results_scores.append(trial_results["scores"])
                hpo_trial_results_imgs_num.append(trial_results["images"])
    if len(hpo_trial_results_scores) != len(hpo_trial_results_imgs_num):
        raise RuntimeError(
            f"mismatch length of trial results and number of trained images {hpo_trial_results_scores}"
            f"/{hpo_trial_results_imgs_num}"
        )
    logger.debug(f"scores = {hpo_trial_results_scores}")
    logger.debug(f"num_imgs = {hpo_trial_results_imgs_num}")
    # Run a SHA (not ASHA)
    rung_score_list = []
    rung_list = _rung_list.copy()
    rung_list.sort(reverse=False)
    logger.debug(f"rung_list = {rung_list}")
    if len(hpo_trial_results_scores) > 1:
        rf = hpo_status["reduction_factor"]

        # for curr_rung_idx, curr_rung in enumerate(rung_list):
        for curr_rung in rung_list:
            if curr_rung <= target_rung:
                rung_score_list.clear()
                logger.debug(f"curr_rung = {curr_rung}")
                for trial, (scores, num_imgs) in enumerate(
                    list(zip(hpo_trial_results_scores, hpo_trial_results_imgs_num))
                ):
                    if num_imgs != []:
                        logger.debug(f"trial{trial} score reported @ {num_imgs}")
                        if num_imgs[-1] > curr_rung:
                            result_idx = -1
                            for img_idx, img in enumerate(num_imgs):
                                if img > curr_rung:
                                    break
                                result_idx = img_idx
                            if result_idx != -1:
                                if result_idx == 0:
                                    rung_score_list.append(scores[result_idx])
                                else:
                                    if mode == "max":
                                        rung_score_list.append(max(scores[:result_idx]))
                                    else:
                                        rung_score_list.append(min(scores[:result_idx]))
                        else:
                            logger.debug(
                                f"cannot find matched result index ({curr_rung}) in {trial} : {num_imgs}. removed"
                            )
                            scores.clear()
                            num_imgs.clear()

                if len(rung_score_list) > 1:
                    logger.debug(f"rung_score_list = {rung_score_list}")
                    if mode == "max":
                        cutt_off_score = np.nanpercentile(
                            rung_score_list, (1 - 1 / rf) * 100
                        )
                    else:
                        cutt_off_score = np.nanpercentile(
                            rung_score_list, (1 / rf) * 100
                        )
                    logger.debug(f"cut_off_score = {cutt_off_score}")
                    for scores, num_imgs in list(
                        zip(hpo_trial_results_scores, hpo_trial_results_imgs_num)
                    ):
                        if num_imgs != []:
                            # if num_imgs[-1] > curr_rung:
                            result_idx = -1
                            for img_idx, img in enumerate(num_imgs):
                                if img > curr_rung:
                                    break
                                result_idx = img_idx
                            if result_idx != -1:
                                if result_idx == 0:
                                    target_score = scores[result_idx]
                                else:
                                    target_score = (
                                        max(scores[:result_idx])
                                        if mode == "max"
                                        else min(scores[:result_idx])
                                    )
                                # logger.debug(f"target score = {target_score}")
                                if mode == "max":
                                    if target_score < cutt_off_score:
                                        scores.clear()
                                        num_imgs.clear()
                                else:
                                    if target_score > cutt_off_score:
                                        scores.clear()
                                        num_imgs.clear()
            # logger.debug(f"rung_score_list = {rung_score_list}")
    if len(rung_score_list) > 1:
        logger.debug(f"last rung_score_list = {rung_score_list}")
        rf = hpo_status["reduction_factor"]

        if mode == "max":
            ret = np.nanpercentile(rung_score_list, (1 - 1 / rf) * 100)
            logger.debug(f"return {ret}")
            return ret
        else:
            ret = np.nanpercentile(rung_score_list, (1 / rf) * 100)
            logger.debug(f"return {ret}")
            return ret

    return None


def report(config: Dict[str, Any], score: float, current_iters: Optional[int] = -1):
    """
    report score to Hpopt.

    Args:
        config (dict): HPO configuration for a trial.
                       This include train confiuration(e.g. hyper parameter, epoch, etc.)
                       and tiral information.
        score (float): score of every epoch during trial.
        current_iters (int): current iteration number when the given score was generated.
    """
    logger.debug(f"report({config})")
    if os.path.exists(config["file_path"]):
        with open(config["file_path"], "rt") as json_file:
            trial_results = json.load(json_file)
    else:
        trial_results = {}
        trial_results["status"] = Status.RUNNING
        trial_results["scores"] = []
        trial_results["median"] = []
        trial_results["images"] = []

    trial_results["scores"].append(score)
    trial_results["median"].append(
        sum(trial_results["scores"]) / len(trial_results["scores"])
    )

    batch_size = config["params"].get(config["batch_size_param_name"])
    logger.debug(f"report() batch size of this trial = {batch_size}")
    if batch_size is None:
        batch_size = config.get("batch_size")
    if batch_size is None:
        raise RuntimeError("cannot find batch size from config or h-params")
    trial_results["images"].append(current_iters * batch_size)

    # Update the current status ASAP in the file system.
    oldmask = os.umask(0o077)
    with open(config["file_path"], "wt") as json_file:
        json.dump(trial_results, json_file, indent=4)
        json_file.flush()
    os.umask(oldmask)

    # if len(trial_results['scores']) >= config["iterations"]:
    if trial_results["images"][-1] >= config["iteration_limit"]:
        trial_results["status"] = Status.STOP
    elif "early_stop" in config and config["early_stop"] == "median_stop":
        save_path = os.path.dirname(config["file_path"])

        median_score = get_median_score(
            save_path, config["trial_id"], len(trial_results["scores"])
        )

        if median_score is not None:
            if config["mode"] == "min":
                curr_best_score = min(trial_results["scores"])
            else:
                curr_best_score = max(trial_results["scores"])

            stop_flag = False

            if config["mode"] == "max" and median_score > curr_best_score:
                stop_flag = True
            elif config["mode"] == "min" and median_score < curr_best_score:
                stop_flag = True

            if stop_flag:
                trial_results["status"] = Status.STOP
                # logger.debug(f"median stop is executed. median score : {median_score} / "
                #              f"current best score : {curr_best_score}")
                logger.debug(
                    f"median stop is executed. median score : {median_score} / "
                    f"current best score : {curr_best_score}"
                )

    elif "rungs" in config:
        # Async HyperBand
        save_path = os.path.dirname(config["file_path"])
        # curr_itr = len(trial_results['scores'])
        curr_itr = trial_results["images"][-1]
        logger.debug(f"current iterations = {curr_itr} : {config['rungs']}")

        for rung_iter_idx, rung_itr in enumerate(config["rungs"]):
            if curr_itr >= rung_itr:
                # Decide whether to promote to the next rung or not
                cutoff_score = get_cutoff_score(
                    save_path, rung_itr, config["rungs"], config["mode"]
                )

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

    oldmask = os.umask(0o077)
    with open(config["file_path"], "wt") as json_file:
        json.dump(trial_results, json_file, indent=4)
        json_file.flush()
    os.umask(oldmask)

    # Wait for flushing updated contents to the file system
    if trial_results["status"] == Status.STOP:
        time.sleep(1)

    return trial_results["status"]


def reportOOM(config):
    """
    report if trial raise out of cuda memory.

    Args:
        config (dict): HPO configuration for a trial.
                       This include train confiuration(e.g. hyper parameter, epoch, etc.)
                       and tiral information.
    """
    trial_results = {}
    trial_results["status"] = Status.CUDAOOM
    trial_results["scores"] = []
    trial_results["median"] = []
    trial_results["images"] = []

    oldmask = os.umask(0o077)
    with open(config["file_path"], "wt") as json_file:
        json.dump(trial_results, json_file, indent=4)
        json_file.flush()
    os.umask(oldmask)

    time.sleep(1)


def load_json(json_file_path: str):
    """
    load json file.

    Args:
        json_file_path (str): path where json file is saved.
    """
    if os.path.exists(json_file_path) is False:
        return None

    retry_flag = True
    contents = None
    while retry_flag:
        try:
            with open(json_file_path, "rt") as json_file:
                contents = json.load(json_file)
            retry_flag = False
        except OSError as err:
            logger.error(f"Error while reading file : {json_file_path}")
            logger.error(f"OS error: {err}")
            return None
        except json.JSONDecodeError:
            time.sleep(0.1)

    return contents


class Status(IntEnum):
    UNKNOWN = -1
    READY = 0
    RUNNING = 1
    STOP = 2
    CUDAOOM = 3
    NORESULT = 4
    PARTIALRESULT = 5
    COMPLETERESULT = 6


class HpoDataset:
    """
    Dataset class which wrap dataset class used in training for sub-sampling.

    Args:
        fullset: dataset instance used in train.
        config (dict): HPO configuration for a trial.
                       This include train confiuration(e.g. hyper parameter, epoch, etc.)
                       and tiral information.
    """

    def __init__(self, fullset, config: Dict[str, Any]):
        self.__dict__ = fullset.__dict__.copy()
        self.fullset = fullset

        if config["subset_ratio"] > 0.0:
            if config["subset_ratio"] < 1.0:
                subset_size = int(len(fullset) * config["subset_ratio"])
                self.subset, _ = random_split(
                    fullset,
                    [subset_size, (len(fullset) - subset_size)],
                    generator=torch.Generator().manual_seed(42),
                )

                # check if fullset is an inheritance of mmdet.datasets
                if hasattr(self, "flag"):
                    self._update_group_flag()
            else:
                self.subset = fullset
            self.length = len(self.subset)
        else:
            self.subset = fullset
            self.length = len(fullset)

        if config["resize_height"] > 0 and config["resize_width"] > 0:
            self.transform = transforms.Resize(
                (config["resize_height"], config["resize_width"]), interpolation=2
            )
        else:
            self.transform = None

    def _update_group_flag(self):
        self.flag = np.zeros(len(self.subset), dtype=np.uint8)

        update_flag = False
        if "img_metas" in self.subset[0]:
            if "ori_shape" in self.subset[0]["img_metas"].data:
                update_flag = True

        if not update_flag:
            return

        for i in range(len(self.subset)):
            self.flag[i] = self.fullset.flag[self.subset.indices[i]]

    def __getitem__(self, index: int):
        data = self.subset[index]
        if self.transform:
            if type(data) is tuple and len(data) == 2 and type(data[0]) == torch.Tensor:
                data = (self.transform(data[0]), data[1])
            elif type(data) is dict and "img" in data:
                data["img"] = self.transform(data["img"])
        return data

    def __len__(self):
        return self.length

    def __getattr__(self, item: str):
        if isinstance(item, str) and (item == "__setstate__" or item == "__getstate__"):
            raise AttributeError(item)

        return getattr(self.fullset, item)


def createHpoDataset(fullset, config: Dict[str, Any]):
    """
    wrap original dataset by HpoDataset using hpo_config returend by Hpopt.

    Args:
        fullset: dataset instance used in train.
        config (dict): HPO configuration for a trial.
                       This include train confiuration(e.g. hyper parameter, epoch, etc.)
                       and tiral information.
    """
    return HpoDataset(fullset, config)
