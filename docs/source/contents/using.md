# Using hpopt

We'll introduce detail of hpopt in this page.
After reading this page, you can get some tips to use hpopt more elegantly.

## Search space

If you want to run HPO, you should have hyper parameters to optmimize.
And you should make hyper parameter search space with that hyper parameters and use it as hpopt aurgment.
Fortunately, it's not hard to make search space.

Search space should be dictionary which has pairs of key and value.
Key is hyper parameter name you can set freely and value is **hpopt.search_space** class.
*search_space* class get two arguments when initiated.
First one is a *type* and second one is a *range*.
*type* is used when sampling hyper parameter from search space.
you can not only sample from normal uniform space, but also sample from quantized search space for some hyper parameter.
For example batch size is generally set as even integer number. In this case, you can use quantized search space.
Another type is log scale search space.
You may want to sample hyper parameter from log scale search space.
For this case, we provide log scale with any base you want.
Of course, you can use log scale quantized search space.
You can also use categorical hyper parameters.
For example if you want to optimize optmizer, you can set this by categorical search space.
With this various search space, you can make any search space you want.
For pratical implementation, refer [](apis)

## hpopt class

After you make search space, you should initiate hpopt class.

Arguments you can set are slightly different depending on whether using baye_opt or ASHA.
But don't worry. They are just advanced options so you can just leave them as default value.
First of all, let's take a look at some important arguments used for initiating class.

There are three required arguments.
- full_dataset_size : It's a just train dataset size.
- num_full_iterations : Epoch for training after HPO.
- search_space : hyper parameter search space to optimize.

You may think that "search_space is ok, but why outhers are required?".
Actually, one magic hides in this.
**The Auto Config.**
HPO finds the optimal hyper-parameters by trying to train the model with various hyper-parameters.
Because of this behavior, it usually takes a huge amount of time to run HPO, which was the biggest obstacle to using HPO practically.
In order to get good results in appropriate time, the user needs to configure the HPO well.
hpopt has a feature called **Auto Config** that configures parameters of HPO automaticaly.
This feature makes hpopt done in expeceted time.
Auto configurated hyper parameters are as below.
- num_trials : Number of times to sample from the hyperparameter search space. It should be greater than or equal to 1.
- max_iterations : Max training epoch for each trials. hpopt will stop training after iterating ‘max_iterations’ times.
- subset_ratio : The ratio of dataset size for HPO task. If this value is greater than or equal to 1.0, full dataset is used for HPO task. When it makes dataset size lower than 500, Dataset size is set to 500.

As you know intuitively, the lower these values are, the faster HPO is.
Auto config is activated when you set *expected_time_ratio* and don't set those arguments.
If you set some of these arugments, then arguments except what you set are configured automatically.
*non_pure_train_ratio* is also used to estimate HPO time more accurately.
Those two things are used for autu configuration to estimate HPO time.
You understand why first two are required now.

```{note}
Expected time ratio means how many times you use for the sum of HPO and finetuning compared to just finetuning time.
In other words, if you set expected time ratio to 2, you'll use same time as training time for HPO.
```

*search_alg* is another important argument. It should be either *baye_opt* or *asha*.
*baye_opt* is sequentail model based optimization(SMBO).
As you can see in the name, *baye_opt* executes each trial sequentailly.
*asha* executes each trial in parallel contrariwise.
So, *baye_opt* is recommeneded if you have few training resources.
Otherwise, *asha* is recommended.

*resume* is convenient feature of hpopt. If you stopped in the middle of HPO for some reason,
and then you want to resume HPO from stopped point, you can do that by set resume to True.
In this case, you should set same *save_path* as you set previously.
Then, hpopt automatically succeeds to previous one.
Another conveninent point is decreasing batch size adaptively.
If you set *batch_size_name* and implement code to report cuda out of memory using hpopt.reportOOM,
hpopt adaptively decrease batch size upper bound and make a new trial.


### Baye_opt (SMBO)

```{image} ./image/smbo.png
:alt: smbo
:scale: 60%
:align: center
:target: https://arxiv.org/pdf/1012.2599v1.pdf
```
*Image by [[Brochu et al., 2010](https://arxiv.org/pdf/1012.2599v1.pdf)]*

Bayesian optimizaiton is one of HPO methods commoly used.
hpopt uses [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) library for SMBO.
It estimates score map from previuos trials using gaussian process,
and chooses next candiate to try.
It's calssic but powerful method.

You can modify behavior of *baye_opt* by giving some arguments. Here is those arguments.
- early_stop : Choice of early stopping methods. Currently ‘median_stop’ is supported.
- num_init_trial : How many trials to use to init SMBO.
- kappa : Kappa vlaue for ucb used in bayesian optimization.
- kappa_decay : Multiply kappa by kappa_decay every trials.
- kappa_decay_delay : Kappa isn’t multiplied to kappa_decay from first trials to kappa_decay_delay trials.

If *early_stop* is set, some trials which is unlikely to get good score stopped in the middle.
It can save time by skipping inefficient trial.
Shortcoming of early stop is that it could skip trial which get lower score early but best score in the end.

To estimate score map, Bayesian optimization needs some initial trial sampled randomly.
*num_init_trial* decides how many trials use random sampled hyper parameters.

Bayesian optmization should balances between exploration and exploitation.
To be more sepecific, exploration means choosing unseen sample from search space
and exploitation means choosing sample which is likely going to provide good score.
If exploration overwhelms, bayesian optimization is almost same as random sampling,
In other case, Many good candiates is left unseen. So it's important to balance between them.
*kappa* is hyper parameter of bayesian optimization.
if higher kappa is higher, bayesian optimization explores more and vice versa.
you can modify this value to change balance.

### ASHA

If you have abundant GPU resouces, it must be better to run HPO parallely.
ASHA is good choice in that case.
ASHA runs multiple trials parallely, compares scores between them and terminates bed trials at specific iteration.
Left trials continue training, are compared again and some of them are terminated at next specific iteration and so on.
It's a like tournament.
Maybe you worry about that late bloomer is terminated eariler.
To avoid that case, ASHA run multiple tournaments some of which has very big interval between checkpoionts.

There are some argument defining ASHA behavior like bayesian optimization.
- min_iterations : hpopt will run training at least ‘min_iterations’ times.
- reduction_factor : Used to set halving rate and amount.
- num_brackets : Number of brackets. Each bracket has a different halving rate, specified by the reduction factor.

*min_iteration* is minimum iterations to compare scores literally. If you want that trial terminated after at least some iterations, you need to set this.

Only 1/n trials can continue training every checkpoint.
*reduction_factor* means "n" used in previuos sentence.
That means if n is high, more of them terminates at checkpoint. It is recommended not to set it too high( > 4).
It could make HPO unstable.

*num_brackets* means number of tournaments. More higher this value is, the more various checkpoint interval there are.


## Implementation Detail

You need to implement loop to run each HPO trial after making hpopt class.
It's totally up to you how to implement the loop.
Although, I provide sample code expecting it could be simple guide to you.

### Implement HPO loop

Let't figure out how to implement loop code.
In this case, I assume that we use *baye_opt* as HPO algorithm.
Because the way of preceeding I'll expalain can be applied to all other environments 
(one node with a GPU, multi node with multi GPU, etc.),
you are able to implement HPO with any environments.

Please take a look at below code.

``` python
import hpopt

def run_hpo_trainer(train_config):
    ...

if __name__ == "__main__":
    hpo = hpopt.create(
        search_alg = "baye_opt",
        ...
    )

    while True:
        train_config = hpo.get_next_sample()
        if train_config is None:
            break
        run_hpo_trainer(train_config)

    best_config = hpo.get_best_config()
    hpo.print_results()
```

That's easy, right?
What you need to do is just getting *train_config* by hpo.get_next_sample() and train model with them.
I skipped how to implement *run_hpo_trainer* function now.
I'll explain in a little while.

Do you remember that hpopt can adaptively decrease batch size if batch size is too big to GPU?
To enable this feature, you should add some code a little bit more. Take a look below code.

``` python
import hpopt

def run_hpo_trainer(train_config):
    ...

if __name__ == "__main__":
    hpo = hpopt.create(
        search_alg = "baye_opt",
        batch_size_name = "bs",
        ...
    )

    while True:
        train_config = hpo.get_next_sample()
        if train_config is None:
            break

        try:
            run_hpo_trainer(train_config)
        # It can be different depending on your training framework.
        except RuntimeError as err: 
            if str(err).startswith("CUDA out of memory"):
                hpopt.reportOOM(train_config)

    best_config = hpo.get_best_config()
    hpo.print_results()
```

You can see that *batch_size_name* is used to initiate hpopt class and
error handling syntax during invoking *run_hpo_trainer*.
If you report "cuda out of memory" situation to hpopt,
hpopt then decrease batch size search space automatically refering *batch_size_name*.


### Implment "run_hpo_trainer" function

It's time to explain how to implement "run_hpo_trainer" function now.
Task of HPO is to optimize objective function which has input as hyper parameter and score as output.
So you need to make function which can get hyper parameters as input, run training with those hyper parameters and return score.
Do you remember "run_hpo_trainer" get **train_config** returned from *hpo.get_best_config()*?
You need to understand what *train_config* is first of all.

**train_config** is dictionary which contains below keys.
- params : hyper parameters to optimize
- iterations : iterations for training
- subset_ratio : train dataset size ratio.

Actually, there are others not written here.
I only introduce keys used in "run_hpo_trainer".
Now, what you need to do is to implement function which train models with these train configurations.
Before I show sample code,
please note that I assume that we use torch as framework,
and of course, you can change this code according to yours.
Let's see below code.

``` python
import torch
import hpopt


def run_hpo_trainer(train_config):
    total_epoch = train_config['iterations']
    lr = train_config['param']['lr']
    bs = train_config['param']['bs']
    subset_ratio = train_config['subset_ratio']

    ...
    train_size = int(subset_ratio * len(full_dataset))
    removed_data = len(full_dataset) - train_size
    training_set, _ = torch.utils.data.random_split(full_dataset, [train_size, removed_data])
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=bs)
    ...
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ...

    def train_one_epoch():
        ...

    for epoch in range(total_epoch):
        model.train(True)
        score = train_one_epoch()
        if hpopt.report(config=hp_config, score=score) == hpopt.Status.STOP:
            break
        ...

if __name__ == "__main__":
    hpo = hpopt.create(...)

    while True:
        train_config = hpo.get_next_sample()
        if train_config is None:
            break
        run_hpo_trainer(train_config)

    best_config = hpo.get_best_config()
    hpo.print_results()
```

You can see that lr and bs are set and training set is splitted according to *subset_ratio*.
After preparation, model is trained with interations given from hpopt.
You shouldn't miss reporting score to hpopt after every epoch.
If hpopt determines that train doesn't need to proceed further,
*hpopt.report()* returns hpopt.Status.STOP.
Then you can just terminate train.

That's it! you can now use hpopt freely according to your need. Please enjoy!
