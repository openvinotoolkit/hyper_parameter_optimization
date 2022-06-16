# Standard


![hpopt_overall](./image/hpopt_overall.png)

- A user specifies the hyper-parameter search space.

- hpopt create a new HpOpt object with the user-provided configurations.

- hpopt tries to find the hyper-parameters that maximizes or minimizes the metric score by training a model with various combinations of hyper-parameters.

- The best hyper-parameters is selected and is used by the following full fine-tuning phase.
