# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import json
import os
import time
from statistics import median

from hpopt.logger import get_logger

logger = get_logger()



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
