# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import pytest
import unittest

from hpopt.base import HpOpt, SearchSpace

# def test_test():
#     os.path.join(None, 'hello.txt')

@pytest.fixture
def int_args():
    return [
        "num_init_trials",
        "num_workers",
        "num_full_iterations",
        "full_dataset_size",
        "min_subset_size"
    ]

@pytest.fixture
def positive_args():
    return [
        "num_init_trials",
        "num_workers",
        "num_full_iterations",
        # "full_dataset_size",
    ]

@pytest.fixture
def optional_int_args():
    return [
        "num_trials",
        "max_iterations",
    ]

@pytest.fixture
def union_float_int_args():
    return [
        "expected_time_ratio",
    ]

@pytest.fixture
def optional_union_float_int_args():
    return [
        "subset_ratio",
    ]


class TestBase:
    @pytest.mark.parametrize("input_val", [ 1.5, -1.0, "string", None])
    def test_init_HpOpt_type_check_int(self, int_args, input_val):
        sp_arg = [SearchSpace('uniform', [0, 1])]

        for arg in int_args:
            kwargs = {arg: input_val}
            print(f"checking '{arg}' with '{input_val}'")
            with pytest.raises(Exception) as e:
                HpOpt(
                    search_space=sp_arg,
                    **kwargs
                )
            print(e)
            assert e.type == TypeError

    @pytest.mark.parametrize("input_val", [ 1.5, -1.0, "string"])
    def test_init_HpOpt_type_check_optinal_int(self, optional_int_args, input_val):
        sp_arg = [SearchSpace('uniform', [0, 1])]

        for arg in optional_int_args:
            kwargs = {arg: input_val}
            print(f"checking '{arg}' with '{input_val}'")
            with pytest.raises(Exception) as e:
                HpOpt(
                    search_space=sp_arg,
                    **kwargs
                )
            print(e)
            assert e.type == TypeError

    @pytest.mark.parametrize("input_val", ["sting", None])
    def test_init_HpOpt_type_check_union_float_int(self, union_float_int_args, input_val):
        sp_arg = [SearchSpace('uniform', [0, 1])]

        for arg in union_float_int_args:
            kwargs = {arg: input_val}
            print(f"checking '{arg}' with '{input_val}'")
            with pytest.raises(Exception) as e:
                HpOpt(
                    search_space=sp_arg,
                    **kwargs
                )
            print(e)
            assert e.type == TypeError

    @pytest.mark.parametrize("input_val", [-1, 0])
    def test_init_HpOpt_type_check_union_float_int(self, positive_args, input_val):
        sp_arg = [SearchSpace('uniform', [0, 1])]

        for arg in positive_args:
            kwargs = {arg: input_val}
            print(f"checking '{arg}' with '{input_val}'")
            with pytest.raises(Exception) as e:
                HpOpt(
                    search_space=sp_arg,
                    **kwargs
                )
            print(e)
            assert e.type == ValueError

    @pytest.mark.parametrize("input_val", ["string"])
    def test_init_HpOpt_type_check_union_float_int(self, optional_union_float_int_args, input_val):
        sp_arg = [SearchSpace('uniform', [0, 1])]

        for arg in optional_union_float_int_args:
            kwargs = {arg: input_val}
            print(f"checking '{arg}' with '{input_val}'")
            with pytest.raises(Exception) as e:
                HpOpt(
                    search_space=sp_arg,
                    **kwargs
                )
            print(e)
            assert e.type == TypeError

class TestSearhSpace(unittest.TestCase):
    def test_search_space(self):
        hp_configs = {
            'a': SearchSpace("uniform", [-5, 10]),
            'b': SearchSpace("quniform", [2, 14, 2]),
            'c': SearchSpace("loguniform", [0.0001, 0.1]),
            'd': SearchSpace("qloguniform", [2, 256, 2]),
            'e': SearchSpace("choice", [98, 765, 4, 321])}
        
        assert len(hp_configs) == 5

        assert hp_configs['a'].space_to_real(hp_configs['a'].lower_space()) == -5
        assert hp_configs['a'].space_to_real(hp_configs['a'].upper_space()) == 10

        assert hp_configs['b'].space_to_real(hp_configs['b'].lower_space()) == 2
        assert hp_configs['b'].space_to_real(hp_configs['b'].upper_space()) == 14

        assert round(hp_configs['c'].space_to_real((hp_configs['c'].lower_space())), 4) == 0.0001
        assert round(hp_configs['c'].space_to_real((hp_configs['c'].upper_space())), 1) == 0.1

        assert round(hp_configs['d'].space_to_real((hp_configs['d'].lower_space())), 0) == 2
        assert round(hp_configs['d'].space_to_real((hp_configs['d'].upper_space())), 0) == 256

        assert hp_configs['e'].space_to_real(hp_configs['e'].lower_space()) == 98
        assert hp_configs['e'].space_to_real(hp_configs['e'].upper_space()) == 321