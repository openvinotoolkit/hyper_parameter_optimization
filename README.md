# [HpOpt] Hyper-parameter Optimization

Python library of automatic hyper-parameter optimization

## Design principles

- Easy to use configuration
  - Support for automatic configuration
- Distributed execution on multi-node
  - Communication via persistent storage
	  - Support for pause/resume
  - Decide to go or stop in the report method
	  - Making a decision without a central controller
- General-purpose design
  - Support for sequential/parallel execution

## Quick Start

- Installation
    ```bash
    $ pip install -r requirements.txt
    $ pip install -v -e .
    ```

- Sample codes
  - List of sample codes
    - Sequential Execution
      - [toy_test.py](../samples/toy_test.py) 
      - [pytorch_test.py](../samples/pytorch_test.py)

    - Parallel Execution
      - [toy_test_parallel.py](../samples/toy_test_parallel.py)
      - [pytorch_parallel_test.py](../samples/pytorch_parallel_test.py)

  - How to run sample codes
      - Basic code to find an optimal point of [skopt.benchmarks.branin](https://scikit-optimize.github.io/stable/modules/generated/skopt.benchmarks.branin.html)
        ```bash
        $ cd samples
        $ pip install -r requirements.txt
        $ python toy_test.py
        ```
        
<!--
## Required arguments for Model Template

- trainer
    - how to run training task for HPO
- hyperparams
    - list of hyperparameter's name, type, and range
- metric: [name of metric to maximize]
- num_trials: [number of maximum trials to run]
- max_iterations: [number of maximum iterations/epochs for each trial]
- subset_ratio: [ratio of items in datasets for HPO training task]
- image_resize: [width and height of images in datasets for HPO training task]
-->
## [[API References]](./docs/apis.md#api)
