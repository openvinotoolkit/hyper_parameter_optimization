# HpOpt API Reference

<!--
#3: [Sonoma Creek](https://gitlab.devtools.intel.com/vmc-eip/IMPT/impt/-/blob/amarek-arch/docs/architecture/platform.md#future-hyper-parameter-optimization)

```python
# train.py
def my_model(x, a, b):
    return a * (x ** 0.5) + b

def my_trainer(config):
    # config (dict): A dict of hyperparameters.
    # common keys
    #   save_path = history save location

    trainset = hpopt.createHpoDataset(trainset, config)

    for iteration_num in config["iterations"]:
        accuracy_score = my_model(iteration_num, config["lr"], config["bs"])
        if hpopt.report(config=config, result=accuracy_score) == hpopt.status.STOP:
            break
```
```python
# HyParOpt operator
hp_configs = {"lr": hpopt.search_space("loguniform", [0.0001, 0.1]),
              "bs": hpopt.search_space("qloguniform", [8, 256, 4]),

# Run 12 trial, stop when trial has reached 10 iterations
my_hpo = hpopt.create(save_path=file_path,
                      search_alg="bayes_opt",
                      search_space=hp_configs,
                      ealry_stop="median_stop",
                      num_init_trials=5,
                      num_trials=12,
                      max_iterations=10,
                      resume=True)

while True:
    config = my_hpo.get_next_sample()

    if config is None:
        break

    # Request a training job with new config
    IMPP.create_training_crd(config)

IMPP.update_hyparopt_crd(my_hpo.get_best_config(wait=True))
```
-->
| `hpopt.create` (***full_dataset_size, num_full_iterations, search_space, save_path='hpo', search_alg='bayes_opt', early_stop=None, mode='max', num_init_trials=5, num_trials=None, max_iterations=None, min_iterations=None, reduction_factor=2, num_brackets=None, subset_ratio=None, batch_size_name=None, image_resize=[0, 0], metric='mAP', resume=False, expected_time_ratio=4, non_pure_train_ratio=0.2, num_workers=1, kappa=2.576, kappa_decay=1, kappa_decay_delay***) |
|--------------|

Initiates a HPO task. If 'num_trials' or 'max_iterations' or 'subset_ratio' are not set, hpopt can find proper values for them which are limited by 'expected_time_ratio'.

- Parameters

  - ***full_dataset_size*** (*int*, `required`) - Train dataset size.

  - ***num_full_iterations*** (*int*, `required`) - Epoch for traninig after HPO.

  - ***search_space*** (*Dict[search_space]*, `required`) – Hyper parameter search space to find.

  - ***save_path*** (*str*, *default="hpo"*)) - Path where result of HPO is saved.

  - ***search_alg*** (*str*, *default=baye_opt*)) - Must be one of ['baye_opt', 'asha'].
                                           Search algorithm to use for optimizing hyper parmaeters.

  - ***early_stop*** (*str*, *default=None*) - Only for SMBO. Choice of early stopping methods. Currently *'median_stop'* is supported.

  - ***mode*** (*str*, *default='max'*) - Must be one of ['min', 'max'].
                                          Determines whether objective is minimizing or maximizing the metric attribute.

  - ***num_init_trials*** (*int*, *default=5*) - Only for SMBO. How many trials to use to init SMBO.

  - ***num_trials*** (*int*, *default=None*) – Number of times to sample from the hyperparameter space.
                                               It should be greater than or equal to 1.
                                               If set to None, it's set automatically.

  - ***max_iterations*** (*int*, *default=None*) - Max training epoch for each trials.
                                                   hpopt will stop training after iterating 'max_iterations' times.

  - ***min_iterations*** (*int*, *default=None*) - Only for ASHA. hpopt will run training at least 'min_iterations' times.

  - ***reduction_factor*** (*int*, *default=2*) - Only for ASHA. Used to set halving rate and amount.

  - ***num_brackets*** (*int*, *default=None*) - Only for ASHA. Number of brackets.
                                                 Each bracket has a different halving rate, specified by the reduction factor.

  - ***subset_ratio*** (*Union[int, float]*, *default=None*) - The ratio of dataset size for HPO task.
                                                               If this value is greater than or equal to 1.0,
                                                               full dataset is used for HPO task.
                                                               When it makes dataset size lower than 500, Dataset size is set to 500.

  - ***batch_size_name*** (*str*, *default=None*) - This is used for CUDA out of memory situation.
                                                    If it is set and CUDA out of memory occurs,
                                                    hpopt automatically decreases batch size to avoid same situation.

  - ***image_resize*** (*List[int]*, *default=[0, 0]*) - The size of image for this HPO task. It has two numbers for width and height respectively.

  - ***metric*** (*str*, *default='mAP'*) - Metric name for HPO.

  - ***resume*** (*bool*, *default=False*) - If True, HPO task resumes or reuses from the results in ***save_path***.
                                             If False, new HPO task is created and old results in ***save_path*** are deleted.

  - ***expected_time_ratio*** (*Union[int, float]*, *default=4*) - The expected ratio of running time for HPO to full
                                                                   fine-tuning time. If this is 4, it means that HPO takes
                                                                   four times longer than full fine-tuning phase.
                                                                   hpopt refers it when configuring HPO automatically.

  - ***non_pure_train_ratio*** (*float*, *default=0.2*) - The ratio of time excluding training over full fine tuning time.
                                                          It's reffered when HPO is automatically configured.

  - ***num_workers*** (*int*, *default=1*) - The number of parallel workers for a trainning.

  - ***kappa*** (*Union[float, int]*, *default=2.576*) - Only for SMBO. Kappa vlaue for ucb used in bayesian optimization.

  - ***kappa_decay*** (*Union[float, int]*, *default=1*) - Only for SMBO. Multiply kappa by kappa_decay every trials.

  - ***kappa_decay_delay*** (*int*, *default=0*) - Only for SMBO. Kappa isn't multiplied to kappa_decay
                                                   from first trials to kappa_decay_delay trials.

- Returns

  - `HpOpt` class instance


| `hpopt.report` (***config, score***) |
|--------------|

Updates a HPO task state and decides to early-stop or not.

- Parameters

  - ***config*** (*Dict[str, Any]*, `required`) - Train confiuration(e.g. hyper parameter, epoch, etc.) for a trial

  - ***score*** (*float*, `required`) - Score of every iteration during trial.

- Returns

  - `hpo.status` - Must be one of [RUNNING, STOP].


| `hpopt.reportOOM` (***config***) |
|--------------|

Report if trial raise out of CUDA memory.

- Parameters

  - ***config*** (*Dict[str, Any]*, `required`) - Train confiuration(e.g. hyper parameter, epoch, etc.) for a trial


| `hpopt.finalize_trial` (***config***) |
|--------------|

Handles the status of trials that have terminated by unexpected causes.

- Parameters

  - ***config*** (*Dict[str, Any]*, `required`) - Train confiuration(e.g. hyper parameter, epoch, etc.) for a trial


| `hpopt.createHpoDataset` (***dataset, config***) |
|--------------|

Creates a proxy dataset.

- Parameters

  - ***dataset*** (*torch.utils.data.Dataset*, `required`) - Full dataset.

  - ***config*** (*Dict[str, Any]*, `required`) - Train configuration for a trial.
                                                  It also has a key 'subset_ratio', 'resize_height', and 'resize_width'
                                                  to change the number of data and resolution of images.


| class `HpOpt` |
|--------------|

An abstract class that searches next samples using underlying hyper-parameter search algorithms.

- Functions

  - `get_next_sample`() --> dict

    Returns the next sample to try. It returs immediately until ***num_init_trials***. 
    After then, it waits for the completion of the previous trial.

  - `get_next_samples`() --> list of dict

    Returns all the available next sample to try.
    As like `get_next_sample`(), it returs immediately until ***num_init_trials***. 
    After then, it waits for the completion of the previous trial.

  - `get_best_config` () --> dict

    Retrieve the best config.

  - `print_results` () --> None

    Print out all trial configurations and scores to stdout.


| class `hpopt.search_space` (***type, range***) |
|--------------|

Class that implements search space used for HPO.
It supports uniform and quantized uniform with normal and log scale in addition to categorical type.
Quantized type has step which is unit for change.

- Parameters

  - ***type*** (*str*, `required`) - Type of hyper parameter search space used for sampling

  - ***range*** (*List[Union[float, int]]*, `required`) - Range of hyper parameter search space.
                                                          Please refer bellow for a detail description.

- Supported Type of Sampling

  - uniform - List[lower: Union[float, int], upper: Union[float, int]]
    - Sample a float value uniformly between lower and upper.

  - quniform - List[lower: Union[float, int], upper: Union[float, int], q: Union[float, int]]
    - Sample a quantized float value uniformly between lower and upper.
      The value will be quantized, i.e. rounded to an integer increment of q.

  - loguniform List[lower: Union[float, int], upper: Union[float, int], base: Union[float, int] = 10)
    - Sample a float value in different orders of magnitude.

  - qloguniform List[lower: Union[float, int], upper: Union[float, int], q: Union[float, int], base: Union[float, int] = 10]
    - Sample a quantized float value in different orders of magnitude.

  - choice [categories: List[Any]]
    - Sample a categorical value.
