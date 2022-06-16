# Detection with OTE

This tutorial provide you how to use HPO for detection.
We'll optimize learning rate and batch size in this tutorial using SMBO which is common method for HPO.

Let's take a look at how HPO can be executed in OTE step by step.

## 1. Set hpo_config.yaml
Before running HPO, you should configure HPO using **hpo_config.yaml**.
It configures everything HPO module needs from hyper parameters you'll optimize to which algorithm to use for HPO.
For more information, you can refer [](hpo_config).
Actually, there is already hpo_config.yaml file with default value in directory where *template.yaml* resides.

Here is default hpo_config.yaml

```
metric: mAP
search_algorithm: smbo
early_stop: None
hp_space:
  learning_parameters.learning_rate:
    param_type: quniform
    range: 
      - 0.001
      - 0.1
      - 0.001
  learning_parameters.batch_size:
    param_type: qloguniform
    range:
      - 4
      - 8
      - 2
```

We can just use this for HPO, but It seems that there is no need to set learning rate search space such that widely.
So, let's modify configuration slightly now.
```
...
...
...
  learning_parameters.learning_rate:
    param_type: quniform
    range: 
      - 0.001
      - 0.01
      - 0.001
...
...
...
```
It looks good. As you see, you can easily change search space or even hyper parameter to optimize by just modifying hpo_config.yaml.

## 2. Run OTE
Now it's time to run OTE. You can enable HPO by just adding an argument **--enable-hpo** literally.
Oops! I forget that we don't have much time to use for HPO.
It would be enough to use same time as training to execute HPO (default hpo-time-ratio is 4).
In this case, you can just add one argument **--hpo-time-ratio** and set it 2.
It means that twice as much time is used as only training time.
```{note}
You should install OTE detection before running OTE.
```
```
ote train \
    training_extensions/external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml \
    --train-ann-files training_extensions/data/airport/annotation_example_train.json \
    --train-data-roots training_extensions/data/airport/train \
    --val-ann-files training_extensions/data/airport/annotation_example_val.json \
    --val-data-roots training_extensions/data/airport/val \
    --save-model-to training_extensions/best.pth \
    --enable-hpo \
    --hpo-time-ratio 2
```
That's it. Now HPO is automatically set to use twice time.
As you can see, you can just add *--hpo-time-ratio* argument to set how much time to use for HPO.

After HPO, HPO result is printed as bellow. You can see which parameter is chosen by this.

```
|  #  | learning_parameters.learning_rate  | learning_parameters.batch_size  |        score        |
|   1 |                              0.005 |                               8 |  0.6179835230685988 |
|   2 |                              0.005 |                              10 |  0.5103616463460058 |
|   3 |                               0.01 |                              48 |  0.6613380273927504 |
|   4 |                              0.006 |                              16 |  0.4305718659552452 |
|   5 |                              0.002 |                              36 |  0.5250386319610906 |
|   6 |               0.009000000000000001 |                              30 |  0.7129637650433698 |
|   7 |                              0.003 |                              44 |  0.9165404011276648 |
|   8 |                               0.01 |                              42 |  0.8234693648419807 |
|   9 |                              0.006 |                               8 |  0.6856121908185698 |
|  10 |                              0.005 |                              30 |  0.8736238451806234 |
|  11 |                              0.005 |                              44 |  0.7295280110191358 |
|  12 |                              0.004 |                              28 |  0.9178114717365546 |
|  13 |                              0.002 |                              44 |  0.8877622967705454 |
|  14 |                              0.003 |                              28 |  0.6154036330149992 |
|  15 |                              0.002 |                              18 |  0.6622151844713957 |
|  16 |                              0.007 |                              24 |  0.7678530317224548 |
|  17 |                              0.003 |                              22 |  0.8609778838120822 |
|  18 |                              0.004 |                              44 |  0.8966648842324036 |
|  19 |                              0.005 |                              28 |  0.8302362663067738 |
|  20 |                              0.005 |                              36 |  0.5476845047852292 |
|  21 |                              0.003 |                              48 |  0.7321795851958944 |
|  22 |                               0.01 |                              34 |  0.9057167695385367 |
|  23 |                              0.008 |                              34 |  0.5073170938766568 |
|  24 |                              0.006 |                              30 |  0.8442392588571863 |
|  25 |               0.009000000000000001 |                              12 |  0.9121951985354356 |
|  26 |                               0.01 |                              12 |  0.6068545610227464 |
|  27 |                              0.002 |                              10 |   0.701241262917842 |
|  28 |                              0.002 |                              20 |  0.9062640968219959 |
|  29 |                               0.01 |                              60 |  0.5049643461876376 |
Best Hyper-parameters
{'learning_parameters.batch_size': 28, 'learning_parameters.learning_rate': 0.004}
```

Then, model is trained with these parameters HPO found.
Now what you need to do is just waiting until all tasks is done.

## 3. Resume & Reuse HPO
You may want to resume HPO after HPO stopped in the middle or reuse hyper parameters HPO found before.
In this case, you can just run OTE same as above.
Then, HPO module automatically searches HPO progress directory(named HPO) used before,
and HPO asks
1. resume in case that HPO is stopped in the middle
2. reuse optimized hyper parameter in case that HPO is finished before
3. rerun HPO from scratch

This is convenient feature to save time by skipping unnecessary HPO.
```{warning}
HPO module asks to you only if HPO configurations are set equivalently as before.
When configurations is changed, HPO is excuted without asking.
```

## 4. (Optional) Hyper parameterse analysis
HPO module saves progress of each trials. So you can use it to analyze trend of hyper parameters by yourself.
It's saved in directory named **hpo** in same path where final model weight is saved.

You can see several files like "hpopt_status.json" and "hpopt_trial_#.json" in the directory.
Someone who are clever may already have notices,
first one conatins overall information of HPO progress and second one contains each trials' infomation.

Detail explanations are as bellow.
- hpopt_status.json
  - search_algorithm : Which algorithm used for HPO.
  - search_space : Which hyper parameters to optmize and scale and range of each hyper parameters.
  - metric : Metric used during HPO.
  - subset_ratio : Ratio of train dataset size used for HPO.
  - image_resize : Ratio of trainning image used for HPO.
  - early_stop : Early stop methods used for SMBO.
  - max_iterations : Maximum iterations for HPO trials.
  - full_dataset_size : Original train dataset size.
  - config_list : List of each trials' information. Each contains trial id, hyper parameters, status and score.
  - num_gen_config : Total number of trials.
  - best_config_id : Trial id containing best score.
- hpopt_trial_#.json : This file contains status and each iteration's score.

By these information, you can manually decide which hyper parameters to use.
Because these files have json format, you can easily load these files from other application.
