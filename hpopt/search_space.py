import math
from typing import Any, Dict, List, Optional, Union, Tuple

from hpopt.logger import get_logger

logger = get_logger()

AVAILABLE_SEARCH_SPACE_TYPE = ["uniform", "quniform", "loguniform", "qloguniform", "choice"]

def _check_type(
    value: Any,
    available_type: Union[type, Tuple],
    variable_name: Optional[str] = None,
    error_message: Optional[str] = None
):
    """Validate value type and if not, raise error."""
    if not isinstance(value, available_type):
        if not isinstance(available_type, tuple):
            available_type = [available_type]
        if error_message is not None:
            message = error_message
        elif variable_name is not None:
            message = (
                f"{variable_name} should be " +
                " or ".join([str(val) for val in available_type]) +
                f". Current {variable_name} type is {type(value)}"
            )
        else:
            raise TypeError
        raise TypeError(message)


class SingleSearchSpace:
    """
    The class which implements single search space used for HPO.
    This class supports uniform and quantized uniform with normal and log scale
    in addition to categorical type. Quantized type has step which is unit for change.

    Args:
        type (str): type of hyper parameter in search space.
                    supported types: uniform, loguniform, quniform, qloguniform, choice
        min (float or int, optional): upper bounding of search space.
                                      If type isn't choice, this value is required.
        max (float or int, optional): lower bounding of search space
                                      If type isn't choice, this value is required.
        step (int, optional): unit for change. If type is quniform or qloguniform,
                              This value is required.
        log_base (int, optional): base of logarithm. Default value is 2.
        choice_list (list, optional): candidiates for choice type. If task is choice,
                                      this value is required.
    """

    def __init__(
        self,
        type: str,
        min: Optional[Union[float, int]] = None,
        max: Optional[Union[float, int]] = None,
        step: Optional[int] = None,
        log_base: Optional[int] = 2,
        choice_list: Optional[Union[List, Tuple]] = None
    ):
        self.type = type
        self.min = min
        self.max = max
        self.step = step
        self.log_base = log_base
        self.choice_list = choice_list

        if self.type not in AVAILABLE_SEARCH_SPACE_TYPE:
            raise ValueError(
                f"type should be one of {', '.join(AVAILABLE_SEARCH_SPACE_TYPE)}. "
                f"But your argument is {self.type}"
            )

        if self.is_categorical():
            _check_type(self.choice_list, (list, tuple), "choice_list")
            if not self.choice_list:
                raise ValueError("If type is choice, choice_list shouldn't be empty.")
            self.choice_list = choice_list
            self.min = 0
            self.max = len(self.choice_list) - 1
        else:
            if min is None:
                raise ValueError("If type isn't choice, you should set min value of search space.")
            if max is None:
                raise ValueError("If type isn't choice, you should set max value of search space.")

            _check_type(self.min, (int, float), "min")
            _check_type(self.max, (int, float), "max")

            if self.use_log_scale():
                _check_type(self.log_base, int, "log_base")
            if self.use_quantized_step():
                if self.step is None:
                    raise ValueError(
                        f"The {self.type} type requires step value. But it doesn't exists"
                    )
                else:
                    _check_type(self.step, int, "step")

    def __repr__(self):
        if self.is_categorical():
            return f"type: {self.type}, candidiate : {', '.join(self.choice_list)}"
        else:
            rep = f"type: {self.type}, search space : {self.min} ~ {self.max}"
            if self.use_quantized_step():
                rep += f", step : {self.step}"
            if self.use_log_scale():
                rep += f", log base : {self.log_base}"
            return  rep

    def is_categorical(self):
        return self.type == "choice"

    def use_quantized_step(self):
        return self.type == "quniform" or self.type == "qloguniform"

    def use_log_scale(self):
        return self.type == "loguniform" or self.type == "qloguniform"

    def lower_space(self):
        if self.use_log_scale():
            return math.log(self.min, self.log_base)
        return self.min

    def upper_space(self):
        if self.use_log_scale():
            return math.log(self.max, self.log_base)
        return self.max

    def space_to_real(self, number: Union[int, float]):
        if self.is_categorical():
            idx = max(min(int(idx), len(self.choice_list) - 1), 0)
            return self.choice_list[idx]
        else:
            if self.use_quantized_step():
                number =  round(number / self.step) * self.step
            if self.use_log_scale():
                number = self.log_base ** number
            return number

    def real_to_space(self, number: Union[int, float]):
        if self.use_log_scale():
            return math.log(number, self.log_base)
        return number

class SearchSpace:
    """
    Class which manages all search spaces of hyper parameters to optimize.
    This class supports HPO algorithms by providing necessary functionalities.

    Args:
        search_space (dict) :
            search spaces of hyper parameters to optimize.
            arguemnt format is as bellow.
            {
                "some_hyper_parameter_name" : {
                    "param_type": type of search space of hyper parameter.
                                  supported types: uniform, loguniform, quniform,
                                  qloguniform or choice
                    # At this point, there are two available formats.
                    # first format is adding each key, value pairs depending on type as bellow.
                    "max" : upper_bound_value,
                    "min" : lower_bound_value,
                    ...
                    # available keys are max, min, step, log_base, choice_list.
                    # second format is adding a list whose key is "range" which contains 
                    # necessary values mentioned above with proper order as bellow.
                    "range" (list): range of hyper parameter search space.
                                    What value at each position means is as bellow.
                                    uniform: [min, max]
                                    quniform: [min, max, step]
                                    loguniform: [min, max, log_base]
                                    qloguniform: [min, max, step, log_base]
                                    choice: [each categorical values, ...]
                    # First format is recommaneded, but you can use any format.
                    # Please refer SingleSearchSpace class comment for
                    # detailed explanation of each vlaues.
                }
                "another_hyper_parameter_name" : {...}
                ...
            }
    """

    def __init__(
        self,
        search_space: Dict[str, Dict[str, Any]],
    ):
        self.search_space = {}

        _check_type(search_space, dict, "search_space")
        for key, val in search_space.items():
            _check_type(val, dict, error_message="Each value of search space should be dict.")
            if "range" not in val:
                self.search_space[key] = SingleSearchSpace(**val)
            else:
                args = {"type" : val["param_type"]}
                if val["param_type"] == "choice":
                    args["choice_list"] = val["range"]
                else:
                    try:
                        args["min"] = val["range"][0]
                        args["max"] = val["range"][1]
                        if args["type"] == "quniform":
                            args["step"] = val["range"][2]
                        elif args["type"] == "loguniform":
                            if len(val["range"]) == 3:
                                args["log_base"] = val["range"][2]
                        elif args["type"] == "qloguniform":
                            args["step"] = val["range"][2]
                            if len(val["range"]) == 4:
                                args["log_base"] = val["range"][3]
                    except IndexError as e:
                        raise e(
                            "You should give all necessary value depending on search space type."
                            "which values are needed depending on type are as bellow."
                            "   - uniform : min value, max value"
                            "   - quniform : min value, max value, step"
                            "   - loguniform : min value, max value, log base(default 2)"
                            "   - qloguniform : min value, max value, step, log baes(default 2)"
                            "But your value is:"
                            f"  - {val['param_type']} : {', '.join(val['range'])}"
                        )
                self.search_space[key] = SingleSearchSpace(**args)

    def __getitem__(self, key):
        return self.search_space[key]

    def __repr__(self):
        return "\n".join(f"{key} => {val}" for key, val in self.search_space.items())

    def __iter__(self):
        return self._search_space_generator()

    def __len__(self):
        return len(self.search_space)

    def _search_space_generator(self):
        for key in self.search_space:
            yield key

    def has_categorical_param(self):
        """check there is a search space whose type is choice"""
        for param in self.search_space.values():
            if param.is_categorical():
                return True
        return False

    def get_real_config(self, config):
        real_config = {}
        for param, value in config.items():
            real_config[param] = self.search_space[param].space_to_real(value)
        return real_config

    def get_space_config(self, config):
        space_config = {}
        for param, value in config.items():
            space_config[param] = self.search_space[param].real_to_space(value)
        return space_config

    def get_bayeopt_search_space(self):
        """return hyper parameter serach sapce as bayeopt library format"""
        bayesopt_space = {}
        for key, val in self.search_space.items():
            bayesopt_space[key] = (
                val.lower_space(),
                val.upper_space()
            )

        return bayesopt_space