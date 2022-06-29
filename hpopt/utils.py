# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
""" collection of the utility functions for the hpopt
"""

import inspect
import json
import os
from typing import Any, Dict, ItemsView, List, Optional, Type, Union

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import transforms

from hpopt.logger import get_logger

logger = get_logger()


def dump_as_json(output_path: str, obj, indent: int = 4):
    """write obj to file as a JSON format"""
    oldmask = os.umask(0o077)
    with open(output_path, "wt", encoding="utf-8") as json_file:
        json.dump(obj, json_file, indent=indent)
        json_file.flush()
    os.umask(oldmask)


def type_check(
    _input: Any,
    expected_type: Union[Type, List[Type]],
    non_zero: Optional[bool] = None,
    positive: Optional[bool] = None,
):
    """helper function to check input paramter varification"""
    callers_local_vars = None
    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        callers_local_vars = frame.f_back.f_locals.items()

    name_of_input = "input"
    if isinstance(callers_local_vars, ItemsView):
        name_list = [var_name for var_name, var_val in callers_local_vars if id(var_val) == id(input)]
        name_of_input = name_list[0] if len(name_list) > 0 and _input is not None else "input"

    allowed = False
    if isinstance(expected_type, list):
        for _type in expected_type:
            if isinstance(_input, _type):
                allowed = True
                break
    elif isinstance(_input, expected_type):
        allowed = True

    if not allowed:
        raise TypeError(
            f"expected type(s) of the '{name_of_input}': {expected_type}, but the input type is {type(_input)}"
        )

    if positive is not None and input is not None:
        if positive and _input < 0:
            raise ValueError(f"'{name_of_input}' is expected to be positive number but {_input}.")
        if not positive and _input > 0:
            raise ValueError(f"'{name_of_input}' is expected to be negative number but {_input}.")

    if non_zero is not None and _input is not None:
        if non_zero and _input == 0:
            raise ValueError(f"'{name_of_input}' should not be zero.")
    return True


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
            self.transform = transforms.Resize((config["resize_height"], config["resize_width"]), interpolation=2)
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
            # if type(data) is tuple and len(data) == 2 and type(data[0]) == torch.Tensor:
            if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], torch.Tensor):
                data = (self.transform(data[0]), data[1])
            # elif type(data) is dict and "img" in data:
            elif isinstance(data, dict) and "img" in data:
                data["img"] = self.transform(data["img"])
        return data

    def __len__(self):
        return self.length

    def __getattr__(self, item: str):
        if isinstance(item, str) and (item in ("__setstate__", "__getstate__")):
            raise AttributeError(item)

        return getattr(self.fullset, item)
