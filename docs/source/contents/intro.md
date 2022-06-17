# What is hpopt?

![hpopt](./image/hpopt.png)

hpopt is a Python library for automatic hyper-parameter optimization, which finds optimized hyper-parameters work best for given tasks. hpopt aims fast hyper-parameter optimization with some convenient features like auto-config.

Hyper-parameter optimization is time-consuming process even with state-of-the-art off-the-shelf libraries. hpopt adopts proxy task, early stopping, and multi-band optimization in a single framework, resulting in several times faster than conventional hyper-parameter optimization. Key features include:

<!-- - **Fast run** : hpopt runs several times faster than other off-the-shelf HPO libraries thanks to a smart combination of data proxy, early stopping, and multi-band optimization. -->

- **Easy of usability** : hpopt takes only time as control parameter, which we believe the most intuitive to most users. auto-config sets internal control parameters automatically with the given time constraint, and guarantees HPO will finish in the constraint.

- **Free from interrupt** : hpopt communicates via persistent storage. Using this, hpopt supports pause/resume/reuse. hpopt always checks that there are previous successful HPO runs and ask you about reusing optimized hyper parameters.

- **Scalability** : hpopt provides both sequential and parallel methods. You can select one depending on your training environment. If you have multiple GPUs, you can accelerate HPO utilizing all of GPU resources.


<!-- hpopt is Python library for automatic hyper-parameter optimization.
hpopt find optimized hyper parameters fitting to current task,
which remove work for user to find good hyper parameters manually. 
hpopt is easy to use and provides convenient features such as *auto config* and *pause/resume*.

hpopt consider time usage for HPO in particular.
HPO is very time consuming tasks.
Someone uses more than month to optimize model hyper parameters with large GPU resources.
But a lot of people are hard to do that.
So we provides some methods like sub dataset or early stopping methods to save time.

In addition to time usage, there are many hpopt's core feature.
Here is hpopt's core features.

- **Auto configuration fitting to desired time** : As we said, HPO is time consuming task. But you don't have to worry about that. If you set desired HPO time, hpopt automatically set these configurations.

- **Distributed execution on multi-node** : hpopt communicates via persistent storage. Using this, hpopt supports pause/resume and reuse. If you want to stop HPO and resume later, you can do that. Also, if hpopt finds that HPO already was done, hpopt asks you about reusing optmized hyper parameters to train.

- **General-purpose design** : hpopt provides both sequantial and parallel methods. You just select one of them depending on your training environment. If you have multi GPU, you can accelerate HPO utilizing all of GPU resource. -->
