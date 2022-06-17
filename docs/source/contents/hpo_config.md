# How to write hpo_config.yaml

HPO is configured by reading **hpo_config.yaml**
-- a yaml file that contains everything related to HPO including hyper prameters to optimize, HPO algorithm, etc..
*hpo_config.yaml* already exists with default value in same directory where *template.yaml* which is essential to run ote resides.
So, you can modify how HPO will run based on this file.

Here is *hpo_config.yaml* provided for classification.

```
metric: mAP
search_algorithm: smbo
early_stop: median_stop
hp_space:
  learning_parameters.learning_rate:
    param_type: quniform
    range: 
      - 0.001
      - 0.01
      - 0.001
  learning_parameters.batch_size:
    param_type: qloguniform
    range: 
      - 8
      - 64
      - 2
```
As you can see, There are some attributes needed to run HPO.
Fortunately, there is not much attribute. It's not difficulty to write your own file.
More detailed description is as bellow.

## Attribute
- **hp_space** (*List[Dict[str, Any]]*, `required`) - Hyper parameter search space to find. It should be list of dictionary. Each has hyper parameter name as key and param_type and range as vaule.
  - ***available hyper parameters***
    - avaiable to both classification and detection
      - learning_parameters.learninig_rate
      - learning_parameters.batch_size
    - available to detection
      - learning_parameters.learning_rate_warmup_iters
  - ***Keys of each hyper parameter***
    - ***param_type*** (*str*, `required`) : Hyper parameter search space type. Must be one of bellows.
      - uniform : Sample a float value uniformly between lower and upper.
      - quniform : Sample a quantized float value uniformly between lower and upper.
      - loguniform : Sample a float value after scaling search space by log scale.
      - qloguniform : Sample a quantized float value after scaling search space by log scale.
      - choice : Sample a categorical value.
    - ***range*** (*List[Any]*, `required`) : Each *param_type* has a respective format.
      - uniform : List[Union[float, int]]
        - lower (*Union[float, int]*, `required`) : lower bound of search space.
        - upper (*Union[float, int]*, `required`) : upper bound of search space.
      - quniform : List[Union[float, int]]
        - lower (*Union[float, int]*, `required`) : lower bound of search space.
        - upper (*Union[float, int]*, `required`) : upper bound of search space.
        - q (*Union[float, int]*, `required`) : unit value of search space.
      - loguniform : List[Union[float, int])
        - lower (*Union[float, int]*, `required`) : lower bound of search space.
        - upper (*Union[float, int]*, `required`) : upper bound of search space.
        - base (*Union[float, int]*, *default=10*) : The logarithm base.
      - qloguniform : List[Union[float, int]]
        - lower (*Union[float, int]*, `required`) : lower bound of search space
        - upper (*Union[float, int]*, `required`) : upper bound of search space
        - q (*Union[float, int]*, `required`) : unit value of search space
        - base (*Union[float, int]*, *default=10*) : The logarithm base.
      - choice : List[Any]
        - vaule : value to be chosen from candidates.
- **metric** (*str*, *default='mAP*') - Metric name for HPO.
- **serach_algorhtim** (*str*, *default=‘baye_opt’*) - Must be one of [‘baye_opt’, ‘asha’]. Search algorithm used for optimizing hyper parmaeters.
- **early_stop** (*str*, *default=None*) - Choice of early stopping methods. Currently ‘median_stop’ is supported.
- **max_iterations** (*int*, *default=None*) - Max training epoch for each trials. hpopt will stop training after iterating ‘max_iterations’ times.
- **subset_ratio** (*Union[float, int]*, *default=None*) - The ratio of dataset size for HPO task. If this value is greater than or equal to 1.0, full dataset is used for HPO task. When it makes dataset size lower than 500, Dataset size is set to 500.
- **num_init_trials** (*int*, *default=5*) - Only for SMBO. How many trials to use to init SMBO.
- **num_trials** (*int*, *default=None*) - Number of times to sample from the hyperparameter space. It should be greater than or equal to 1. If set to None, it’s set automatically.
- **num_brackets** (*int*, *default=None*) - Only for ASHA. Number of brackets. Each bracket has a different halving rate, specified by the reduction factor.
- **min_iterations** (*int*, *default=None*) - Only for ASHA. hpopt will run training at least ‘min_iterations’ times.
- **reduction_factor** (*float*, *default=2*) - Only for ASHA. Used to set halving rate and amount.