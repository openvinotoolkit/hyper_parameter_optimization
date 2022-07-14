# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import pytest

from hpopt.utils import type_check

def test_utils_type_check():
    assert type_check(1, int) == True
    assert type_check(1, [int]) == True
    assert type_check(1.0, float) == True
    assert type_check(1.0, [float]) == True

    assert type_check(1, [int, float]) == True
    assert type_check(1.0, [int, float]) == True

    assert type_check(None, [int, type(None)]) == True
    assert type_check("hello", [str]) == True
    assert type_check("hello", [str, type(None)]) == True
    assert type_check(None, [str, type(None)]) == True

    assert type_check(-1, int) == True

    assert type_check(1, int, non_zero=True) == True
    assert type_check(10, int, non_zero=True) == True
    assert type_check(-1, int, non_zero=True) == True

    assert type_check(1, int, non_zero=True, positive=True) == True
    assert type_check(10, int, non_zero=True, positive=True) == True
    assert type_check(-1, int, non_zero=True, positive=False) == True

    assert type_check(1, [int, type(None)], non_zero=True, positive=True) == True
    assert type_check(10, [int, type(None)], non_zero=True, positive=True) == True
    assert type_check(None, [int, type(None)], non_zero=True, positive=True) == True

    assert type_check(1, [int, float, type(None)]) == True
    assert type_check(1.0, [int, float, type(None)]) == True
    assert type_check(0, [int, float, type(None)]) == True
    assert type_check(-1.0, [int, float, type(None)]) == True
    assert type_check(-1, [int, float, type(None)]) == True
    assert type_check(None, [int, float, type(None)]) == True

    assert type_check(1.0, [int, float, type(None)], positive=True) == True
    assert type_check(-1.0, [int, float, type(None)], positive=False) == True
    assert type_check(-1, [int, float, type(None)], positive=False) == True
    assert type_check(None, [int, float, type(None)], positive=True) == True

    with pytest.raises(Exception) as e:
        type_check(1, str)
    assert e.type == TypeError

    with pytest.raises(Exception) as e:
        type_check(None, str)
    assert e.type == TypeError

    with pytest.raises(Exception) as e:
        type_check(1.0, int)
    assert e.type == TypeError

    with pytest.raises(Exception) as e:
        type_check(-1.0, int)
    assert e.type == TypeError

    with pytest.raises(Exception) as e:
        type_check(None, int)
    assert e.type == TypeError

    with pytest.raises(Exception) as e:
        type_check(0, float)
    assert e.type == TypeError

    with pytest.raises(Exception) as e:
        type_check(None, float)
    assert e.type == TypeError

    with pytest.raises(Exception) as e:
        type_check("hello", [float, int])
    assert e.type == TypeError

    with pytest.raises(Exception) as e:
        type_check(None, [float, int])
    assert e.type == TypeError

    with pytest.raises(Exception) as e:
        type_check(-1.0, float, positive=True)
    assert e.type == ValueError

    with pytest.raises(Exception) as e:
        type_check(-1, int, positive=True)
    assert e.type == ValueError

    with pytest.raises(Exception) as e:
        type_check(0, int, non_zero=True)
    assert e.type == ValueError

    with pytest.raises(Exception) as e:
        type_check(-1, int, non_zero=True, positive=True)
    assert e.type == ValueError

    with pytest.raises(Exception) as e:
        assert type_check(1, int, non_zero=True, positive=False)
    assert e.type == ValueError

    with pytest.raises(Exception) as e:
        type_check(10, int, non_zero=True, positive=False)
    assert e.type == ValueError

    with pytest.raises(Exception) as e:
        type_check(-1, [int, type(None)], non_zero=True, positive=True)
    assert e.type == ValueError

    with pytest.raises(Exception) as e:
        type_check(0, [int, float, type(None)], positive=True, non_zero=True)
    assert e.type == ValueError

    my_variable_name_strange = 1.0
    with pytest.raises(Exception) as e:
        type_check(my_variable_name_strange, int)
    assert "my_variable_name_strange" in str(e.value)
