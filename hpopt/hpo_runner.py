import multiprocessing
import os
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, Optional, Union

from hpopt.hpo_base import HpoBase, Trial
from hpopt.logger import get_logger

try:
    import pynvml
except ImportError:
    pynvml = None

logger = get_logger()


class ResourceManager(ABC):
    @abstractmethod
    def reserve_resource(self, trial_id):
        raise NotImplementedError

    @abstractmethod
    def release_resource(self, trial_id):
        raise NotImplementedError

    @abstractmethod
    def have_available_resource(self):
        raise NotImplementedError

class CPUResourceManager(ResourceManager):
    def __init__(self, num_parallel_trial: int = 4):
        self._num_parallel_trial = num_parallel_trial
        self._usage_status = []

    def reserve_resource(self, trial_id: Any):
        if not self.have_available_resource():
            return None
        if trial_id in self._usage_status:
            raise RuntimeError(f"{trial_id} already has reserved resource.")

        logger.debug(f"{trial_id} reserved.")
        self._usage_status.append(trial_id)
        return {}

    def release_resource(self, trial_id: Any):
        if trial_id not in self._usage_status:
            logger.warning(f"{trial_id} trial don't use resource now.")
        else:
            self._usage_status.remove(trial_id)
            logger.debug(f"{trial_id} released.")

    def have_available_resource(self):
        return len(self._usage_status) < self._num_parallel_trial

class GPUResourceManager(ResourceManager):
    def __init__(self, num_gpu_for_single_trial: int = 1, available_gpu: Optional[str] = None):
        self._num_gpu_for_single_trial = num_gpu_for_single_trial
        self._usage_status = {}
        if self._available_gpu is None:
            num_gpus = pynvml.nvmlDeviceGetCount()
            self._available_gpu = [val for val in range(num_gpus)]
        else:
            self._available_gpu = [int(val) for val in available_gpu.split(',')]

    def reserve_resource(self, trial_id):
        if not self.have_available_resource():
            return None
        if trial_id in self._usage_status:
            raise RuntimeError(f"{trial_id} already has reserved resource.")

        resource = list(self._available_gpu[:self._num_gpu_for_single_trial])
        self._available_gpu = self._available_gpu[self._num_gpu_for_single_trial:]

        self._usage_status[trial_id] = resource
        return {"CUDA_VISIBLE_DEVICES" : ",".join([str(val) for val in resource])}


    def release_resource(self, trial_id):
        if trial_id not in self._usage_status:
            logger.warning(f"{trial_id} trial don't use resource now.")
        else:
            self._available_gpu.extend(self._usage_status[trial_id])
            del self._usage_status[trial_id]

    def have_available_resource(self):
        return len(self._available_gpu) >= self._num_gpu_for_single_trial

def get_resource_manager(
    resource_type: str,
    num_parallel_trial: Optional[int] = None,
    num_gpu_for_single_trial: Optional[int] = None,
    available_gpu: Optional[str] = None,
):
    if resource_type == "cpu":
        args = {"num_parallel_trial" : num_parallel_trial}
        args = _remove_none_from_dict(args)
        return CPUResourceManager(**args)
    elif resource_type == "gpu":
        args = {"num_gpu_for_single_trial" : num_gpu_for_single_trial, "available_gpu" : available_gpu}
        args = _remove_none_from_dict(args)
        return GPUResourceManager(**args)
    else:
        raise ValueError(f"Available resource type is cpu, gpu. Your value is {resource_type}.")

def _remove_none_from_dict(d: Dict):
    key_to_remove = [key for key, val in d.items() if val is None] 
    for key in key_to_remove:
        del d[key]
    return d

class HpoLoop:
    def __init__(self, hpo_algo: HpoBase, train_func: Callable, resource_manager: Optional[ResourceManager] = None):
        self._hpo_algo = hpo_algo
        self._train_func = train_func
        self._running_trials: Dict[int, Process] = {}
        self._mp = multiprocessing.get_context("spawn")
        self._report_queue = self._mp.Queue()
        self._uid_index = 0
        self._resource_manager = resource_manager
        if self._resource_manager is None:
            self._resource_manager = get_resource_manager("cpu")

    def run(self):
        logger.info("HPO loop starts.")
        while not self._hpo_algo.is_done():
            if self._resource_manager.have_available_resource():
                trial = self._hpo_algo.get_next_sample()
                if trial is not None:
                    self._start_trial_process(trial)

            self._remove_finished_process()
            self._get_reports()

        logger.info("HPO loop is done.")
        self._get_reports()
        self._join_all_processes()

        return self._hpo_algo.get_best_config()

    def _start_trial_process(self, trial: Trial):
        logger.info(f"{trial.id} trial is now running.")
        logger.debug(f"{trial.id} hyper paramter => {trial.configuration}")
        
        uid = self._get_uid()
        env = self._resource_manager.reserve_resource(uid)
        process = self._mp.Process(
            target=_run_train,
            args=(
                self._train_func,
                trial.configuration,
                partial(_report_score, report_queue=self._report_queue, trial_id=trial.id),
                env
            )
        )
        self._running_trials[uid] = process
        process.start()

    def _remove_finished_process(self):
        trial_to_remove = []
        for uid, process in self._running_trials.items():
            if not process.is_alive():
                process.join()
                trial_to_remove.append(uid)

        for uid in trial_to_remove:
            self._resource_manager.release_resource(uid)
            del self._running_trials[uid]

    def _get_reports(self):
        while not self._report_queue.empty():
            report = self._report_queue.get(timeout=3)
            self._hpo_algo.report_score(
                report["score"],
                report["progress"],
                report["trial_id"],
            )
    
    def _join_all_processes(self):
        for p in self._running_trials.values():
            p.join()

        self._running_trials = {}

    def _get_uid(self):
        uid = self._uid_index
        self._uid_index += 1
        return uid

def _run_train(train_func: Callable, hp_config: Dict, report_func: Callable, env: Optional[Dict] = None):
    if env is not None:
        for key, val in env.items():
            os.environ[key] = val
    train_func(hp_config, report_func)

def _report_score(
    score: Union[int, float],
    progress: Union[int, float],
    report_queue: Queue,
    trial_id: Any
):
    logger.debug(f"score : {score}, progress : {progress}, trial_id : {trial_id}, pid : {os.getpid()}")
    report_queue.put_nowait(
        {
            "score" : score,
            "progress" : progress,
            "trial_id" : trial_id,
            "pid" : os.getpid()
        }
    )

def run_hpo(
    hpo_algo: HpoBase,
    train_func: Callable,
    resource_type: str = "cpu",
    num_parallel_trial: Optional[int] = None,
    num_gpu_for_single_trial: Optional[int] = None,
    available_gpu: Optional[str] = None,
):
    resource_manager = get_resource_manager(
        resource_type,
        num_gpu_for_single_trial,
        available_gpu,
        num_parallel_trial
    )
    hpo_loop = HpoLoop(hpo_algo, train_func, resource_manager)
    best_config = hpo_loop.run()
    return best_config
