import math
import pytest

from hpopt.search_space import SingleSearchSpace, SearchSpace

ALL_TYPE = ["uniform", "loguniform", "quniform", "qloguniform", "choice"]
NOT_CATEGORICAL_TYPE = ["uniform", "loguniform", "quniform", "qloguniform"]
USE_LOG_SCALE_TYPE = ["loguniform", "qloguniform"]
USE_QUANTIZED_STEP_TYPE = ["quniform", "qloguniform"]

def make_single_search_space_class(
    type="uniform",
    min=2, 
    max=100,
    step=2,
    log_base=2,
    choice_list=[1,2,3]
):
    return SingleSearchSpace(type, min, max, step, log_base, choice_list)

class TestSingleSearchSpace:
    @pytest.mark.parametrize("type", NOT_CATEGORICAL_TYPE)
    @pytest.mark.parametrize("min,max", [(1,100), (1.12, 53.221)])
    @pytest.mark.parametrize("step", [2, 10])
    @pytest.mark.parametrize("log_base", [2, 10, None])
    def test_init_on_not_choice_type_with_good_input(self, type, min, max, step, log_base):
        if log_base is None:
            search_space_class = SingleSearchSpace(type, min, max, step)
        else:
            search_space_class = SingleSearchSpace(type, min, max, step, log_base)

    @pytest.mark.parametrize("choice_list", [("abc", "def"), [1,2,3]])
    def test_init_on_choice_type_with_good_input(self, choice_list):
        search_space_class = SingleSearchSpace("choice", choice_list=choice_list)

    @pytest.mark.parametrize("val", ["string", [1,2], {1:2}])
    def test_init_wrong_min_max_type(self, val):
        # check min
        with pytest.raises(TypeError):
            search_space_class = SingleSearchSpace("uniform", val, 100, 2, 2)
        # check max
        with pytest.raises(TypeError):
            search_space_class = SingleSearchSpace("uniform", 2, val, 2, 2)

    @pytest.mark.parametrize("step", ["string", [1,2], {1:2}])
    def test_init_wrong_step_type(self, step):
        with pytest.raises(TypeError):
            search_space_class = SingleSearchSpace("qloguniform", 2, 100, step, 2)

    @pytest.mark.parametrize("log_base", ["string", [1,2], {1:2}, 1.2])
    def test_init_wrong_log_base_type(self, log_base):
        with pytest.raises(TypeError):
            search_space_class = SingleSearchSpace("qloguniform", 2, 100, 2, log_base)

    @pytest.mark.parametrize("choice_list", [1, 1.2, "atr", {1:2, 2:4}])
    def test_init_wrong_choice_list_type(self, choice_list):
        with pytest.raises(TypeError):
            search_space_class = SingleSearchSpace("choice", choice_list)

    @pytest.mark.parametrize("min,max", [(None, 100), (2, None), (None, None)])
    def test_init_no_min_max_value(self, min, max):
        with pytest.raises(ValueError):
            search_space_class = SingleSearchSpace("uniform", min, max)

    @pytest.mark.parametrize("type", USE_QUANTIZED_STEP_TYPE)
    def test_init_no_step_value(self, type):
        with pytest.raises(ValueError):
            search_space_class = SingleSearchSpace(type, 2, 100, None, 2)

    @pytest.mark.parametrize("min,max", [(3, 1), (5.5, -12.3), (1, 1), (2.0, 2.0)])
    def test_init_min_is_bigger_or_same_than_max(self, min, max):
        with pytest.raises(ValueError):
            search_space_class = SingleSearchSpace("uniform", min, max)

    @pytest.mark.parametrize("type", USE_LOG_SCALE_TYPE)
    @pytest.mark.parametrize("min,max", [(-20, -12), (-12.124, 10), (0, 3)])
    def test_minus_value_in_log_type(self, type, min, max):
        with pytest.raises(ValueError):
            search_space_class = SingleSearchSpace(type, min, max, step=2, log_base=2)


    @pytest.mark.parametrize("type", ["wrong_type", 12, 1.24, [1,2]])
    def test_init_with_wrong_type(self, type):
        with pytest.raises(ValueError):
            search_space_class = SingleSearchSpace(type, 1, 100, 2, 2, [1,2])

    @pytest.mark.parametrize("type,min,max,step", [("qloguniform", 1, 2, 100), ("quniform",-0.1, 0.1, 1)])
    def test_init_step_is_too_big(self, type, min, max, step):
        with pytest.raises(ValueError):
            search_space_class = SingleSearchSpace(type, min, max, step, 2)

    @pytest.mark.parametrize("type", USE_LOG_SCALE_TYPE)
    @pytest.mark.parametrize("log_base", [1, 0, -1])
    def test_init_log_base_is_wrong_value(self, type, log_base):
        with pytest.raises(ValueError):
            search_space_class = SingleSearchSpace(type, 1, 100, 2, log_base)

    @pytest.mark.parametrize("choice_list", [[], [1]])
    def test_init_empty_or_single_element_choice_list(self,choice_list):
        with pytest.raises(ValueError):
            search_space_class = SingleSearchSpace("choice", choice_list=choice_list)

    @pytest.mark.parametrize("type", ALL_TYPE)
    def test_is_categorical(self, type):
        sss = make_single_search_space_class(type)
        if type in NOT_CATEGORICAL_TYPE:
            assert sss.is_categorical() == False
        else:
            assert sss.is_categorical() == True

    @pytest.mark.parametrize("type", ALL_TYPE)
    def test_use_quantized_step(self, type):
        sss = make_single_search_space_class(type)
        if type in USE_QUANTIZED_STEP_TYPE:
            assert sss.use_quantized_step() == True
        else:
            assert sss.use_quantized_step() == False

    @pytest.mark.parametrize("type", ALL_TYPE)
    def test_use_log_scale(self, type):
        sss = make_single_search_space_class(type)
        if type in USE_LOG_SCALE_TYPE:
            assert sss.use_log_scale() == True
        else:
            assert sss.use_log_scale() == False

    @pytest.mark.parametrize("type", ALL_TYPE)
    @pytest.mark.parametrize("min,max", [(10, 100), (12, 30), (124.12, 12122.151)])
    @pytest.mark.parametrize("log_base", [2, 10])
    def test_lower_space_upper_space(self, type, min, max, log_base):
        sss = make_single_search_space_class(type, min=min, max=max, log_base=log_base)
        if type in USE_LOG_SCALE_TYPE:
            assert sss.lower_space() == math.log(min, log_base)
            assert sss.upper_space() == math.log(max, log_base)
        elif type in NOT_CATEGORICAL_TYPE:
            assert sss.lower_space() == sss.min
            assert sss.upper_space() == sss.max
        else:
            assert sss.lower_space() == 0
            assert sss.upper_space() == len(sss.choice_list) - 1

    @pytest.mark.parametrize("type", NOT_CATEGORICAL_TYPE)
    @pytest.mark.parametrize("number", [2.3, 15])
    @pytest.mark.parametrize("step", [2, 3, 7])
    @pytest.mark.parametrize("log_base", [2, 10])
    def test_space_to_real_not_categorical_type(self, type, number, step, log_base):
        sss = make_single_search_space_class(type, step=step, log_base=log_base)
        ret = sss.space_to_real(number)
        expected_ret = number

        if type in USE_LOG_SCALE_TYPE:
            expected_ret = log_base ** expected_ret
        if type in USE_QUANTIZED_STEP_TYPE:
            gap = sss.min % step
            expected_ret = round((expected_ret - gap) / step) * step  + gap

        assert ret == expected_ret

    @pytest.mark.parametrize("number,choice_list", [(2, [1,2,3]), (10, [1,2]), (-3, [1,2])])
    def test_space_to_real_categorical_type(self, number, choice_list):
        sss = make_single_search_space_class("choice", choice_list=choice_list)
        ret = sss.space_to_real(number)
        expected_ret = min(max(int(number), 0), len(choice_list)-1)
        assert ret == sss.choice_list[expected_ret]

    @pytest.mark.parametrize("number", ["string", [1,2], {1:2}])
    def test_space_to_real_wrong_type(self, number):
        sss = make_single_search_space_class()
        with pytest.raises(TypeError):
            sss.real_to_space(number)

    @pytest.mark.parametrize("type", ALL_TYPE)
    @pytest.mark.parametrize("number", [10, 512.3])
    @pytest.mark.parametrize("log_base", [2, 10])
    def test_real_to_space(self, type, log_base, number):
        sss = make_single_search_space_class(type, log_base=log_base)
        if type in USE_LOG_SCALE_TYPE:
            assert sss.real_to_space(number) == math.log(number, log_base)
        else:
            assert sss.real_to_space(number) == number

    @pytest.mark.parametrize("number", ["string", [1,2], {1:2}])
    def test_real_to_space_wrong_type(self, number):
        sss = make_single_search_space_class()
        with pytest.raises(TypeError):
            sss.real_to_space(number)

class TestSearchSpace:
    @staticmethod
    def get_search_space_depending_on_type(types, range_format=False):
        if not isinstance(types, (list, tuple)):
            types = [types]
        search_space = {}

        if "uniform" in types:
            TestSearchSpace.add_uniform_search_space(search_space, range_format)
        if "quniform" in types:
            TestSearchSpace.add_quniform_search_space(search_space, range_format)
        if "loguniform" in types:
            TestSearchSpace.add_loguniform_search_space(search_space, range_format)
        if "qloguniform" in types:
            TestSearchSpace.add_qloguniform_search_space(search_space, range_format)
        if "choice" in types:
            TestSearchSpace.add_choice_search_space(search_space)

        return search_space

    @staticmethod
    def add_uniform_search_space(search_space, range_format=False):
        search_space["uniform_search_space"] = {"param_type" : "uniform"}
        if range_format:
            search_space["uniform_search_space"]["range"] = [1, 10]
        else:
            search_space["uniform_search_space"].update(
                {"min" : 1, "max" : 10}
            )

    @staticmethod
    def add_quniform_search_space(search_space, range_format=False):
        search_space["quniform_search_space"] = {"param_type" : "quniform"}
        if range_format:
            search_space["quniform_search_space"]["range"] = [1, 10, 3]
        else:
            search_space["quniform_search_space"].update(
                {"min" : 1, "max" : 10, "step" : 3}
            )

    @staticmethod
    def add_loguniform_search_space(search_space, range_format=False):
        search_space["loguniform_search_space"] = {"param_type" : "loguniform"}
        if range_format:
            search_space["loguniform_search_space"]["range"] = [1, 10, 2]
        else:
            search_space["loguniform_search_space"].update(
                {"min" : 1, "max" : 10, "log_base" : 2}
            )

    @staticmethod
    def add_qloguniform_search_space(search_space, range_format=False):
        search_space["qloguniform_search_space"] = {"param_type" : "qloguniform"}
        if range_format:
            search_space["qloguniform_search_space"]["range"] = [1, 10, 3, 2]
        else:
            search_space["qloguniform_search_space"].update(
                {"min" : 1, "max" : 10, "step" : 3, "log_base" : 2}
            )

    @staticmethod
    def add_choice_search_space(search_space):
        search_space["choice_search_space"] = {
            "param_type" : "choice",
            "choice_list" : ["somevalue1", "somevalue2", "somevalue3"]
        }

    @pytest.fixture
    def search_space_with_all_types(self):
        return SearchSpace(self.get_search_space_depending_on_type(ALL_TYPE))

    def test_init_with_range_format_argument(self):
        search_space = self.get_search_space_depending_on_type(ALL_TYPE, True)
        ss = SearchSpace(search_space)
        assert ss is not None

    @pytest.mark.parametrize("type", NOT_CATEGORICAL_TYPE)
    def test_init_with_insufficient_range_arguments(self, type):
        search_space = self.get_search_space_depending_on_type(type, True)
        if type in USE_LOG_SCALE_TYPE:
            num_to_delete = 2
        else:
            num_to_delete = 1
        search_space[f"{type}_search_space"]["range"] = search_space[f"{type}_search_space"]["range"][:-num_to_delete]
        with pytest.raises(ValueError):
            ss = SearchSpace(search_space)

    @pytest.mark.parametrize("wrong_type_val", [1, 1.2, "string"])
    def test_init_with_wrong_range_type(self, wrong_type_val):
        search_space = self.get_search_space_depending_on_type(ALL_TYPE, True)
        for val in search_space.values():
            val["range"] = wrong_type_val
        with pytest.raises(TypeError):
            ss = SearchSpace(search_space)

    def test_init_both_format_exists(self):
        search_space = self.get_search_space_depending_on_type(ALL_TYPE)
        range_format = self.get_search_space_depending_on_type(ALL_TYPE, True)
        for key, val in search_space.items():
            val.update(range_format[key])
        ss = SearchSpace(search_space)

    def test_get_item_available(self, search_space_with_all_types):
        for type in ALL_TYPE:
            val = search_space_with_all_types[f"{type}_search_space"]

    def test_iteratble(self, search_space_with_all_types):
        for val in search_space_with_all_types:
            pass

    def test_len_is_available(self, search_space_with_all_types):
        assert len(search_space_with_all_types) == 5

    @pytest.mark.parametrize("choice_exist", [True, False])
    def test_has_categorical_param(self, search_space_with_all_types, choice_exist):
        if choice_exist:
            search_space = self.get_search_space_depending_on_type(ALL_TYPE)
        else:
            search_space = self.get_search_space_depending_on_type(NOT_CATEGORICAL_TYPE)
        ss = SearchSpace(search_space)

        assert ss.has_categorical_param() == choice_exist


    def test_get_real_config_with_proper_argument(self, search_space_with_all_types):
        # search space configuration
        step = 3
        log_base = 2
        min_val = 1
        requested_val = 3.2

        config = {f"{type}_search_space" : requested_val for type in ALL_TYPE}
        real_space = search_space_with_all_types.get_real_config(config)

        for key, val in real_space.items():
            rescaled_requested_val = requested_val
            if key in ["loguniform_search_space", "qloguniform_search_space"]:
                rescaled_requested_val = log_base ** requested_val
            if key in ["quniform_search_space" ,"qloguniform_search_space"]:
                gap = min_val % step
                rescaled_requested_val = round((rescaled_requested_val - gap) / step) * step + gap
            if key == "choice_search_space":
                choice_list = search_space_with_all_types[key].choice_list
                idx = max(min(int(rescaled_requested_val), len(choice_list) - 1), 0)
                rescaled_requested_val = choice_list[idx]

            assert val == rescaled_requested_val

    @pytest.mark.parametrize("wrong_name", ["wrong_name", 1, 3.2])
    def test_get_real_config_with_wrong_name_config(self, search_space_with_all_types, wrong_name):
        config = {wrong_name : 3.2}
        with pytest.raises(KeyError):
            real_space = search_space_with_all_types.get_real_config(config)

    @pytest.mark.parametrize("wrong_value", ["wrong_value", [1,3,4], (1,2)])
    def test_get_real_config_with_wrong_value_config(self, search_space_with_all_types, wrong_value):
        config = {"quniform_search_space" : wrong_value}
        with pytest.raises(TypeError):
            real_space = search_space_with_all_types.get_real_config(config)

    def test_get_space_config_with_proper_argument(self, search_space_with_all_types):
        # search space configuration
        log_base = 2
        requested_val = 10

        config = {f"{type}_search_space" : requested_val for type in ALL_TYPE}
        real_space = search_space_with_all_types.get_space_config(config)

        for key, val in real_space.items():
            rescaled_requested_val = requested_val
            if key in ["loguniform_search_space", "qloguniform_search_space"]:
                rescaled_requested_val = math.log(requested_val, log_base)

            assert val == rescaled_requested_val

    @pytest.mark.parametrize("wrong_name", ["wrong_name", 1, 3.2])
    def test_get_space_config_with_wrong_name_config(self, search_space_with_all_types, wrong_name):
        config = {wrong_name : 3.2}
        with pytest.raises(KeyError):
            real_space = search_space_with_all_types.get_space_config(config)

    @pytest.mark.parametrize("wrong_value", ["wrong_value", [1,3,4], (1,2)])
    def test_get_space_config_with_wrong_value_config(self, search_space_with_all_types, wrong_value):
        config = {"quniform_search_space" : wrong_value}
        with pytest.raises(TypeError):
            real_space = search_space_with_all_types.get_space_config(config)

    def test_get_bayeopt_search_space(self, search_space_with_all_types):
        bayes_opt_format = search_space_with_all_types.get_bayeopt_search_space()

        for val in bayes_opt_format.values():
            assert len(val) == 2
            min_val, max_val = val
            assert min_val < max_val
