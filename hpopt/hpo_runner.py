import os
import multiprocessing
from functools import partial
from multiprocessing import Process, Queue
from typing import Any, Dict, Union, Callable

from hpopt.logger import get_logger
from hpopt.hpo_base import HpoBase, Trial

logger = get_logger()


class HpoLoop:
    def __init__(self, hpo_algo: HpoBase, train_func: Callable):
        self._hpo_algo = hpo_algo
        self._train_func = train_func
        self._processes: Dict[int, Process] = {}
        self._mp = multiprocessing.get_context("spawn")
        self._report_queue = self._mp.Queue()

    def run(self):
        logger.info("HPO loop starts.")
        while not self._hpo_algo.is_done():
            if self._have_resource_to_start_new_process():
                trial = self._hpo_algo.get_next_sample()
                if trial is not None:
                    self._start_trial_process(trial)

            self._remove_finished_process()
            self._get_reports_and_stop_trial_if_necessary()

        logger.info("HPO loop is done.")
        self._get_reports_and_stop_trial_if_necessary()
        self._join_all_processes()

        return self._hpo_algo.get_best_config()

    def _start_trial_process(self, trial: Trial):
        logger.info(f"{trial.id} trial is now running.")
        logger.debug(f"{trial.id} hyper paramter => {trial.configuration}")
        process = self._mp.Process(
            target=self._train_func,
            args=(
                trial.configuration,
                partial(_report_score, report_queue=self._report_queue, trial_id=trial.id)
            )
        )
        self._processes[process.pid] = process
        process.start()

    def _remove_finished_process(self):
        alive_process = self._mp.active_children()
        self._processes = {process.pid : process for process in alive_process}

    def _get_reports_and_stop_trial_if_necessary(self):
        while not self._report_queue.empty():
            report = self._report_queue.get(timeout=3)
            need_to_stop = self._hpo_algo.report_score(
                report["score"],
                report["progress"],
                report["trial_id"],
            )
            if need_to_stop:
                self._processes[report["pid"]].terminate()
    
    def _have_resource_to_start_new_process(self):
        return len(self._processes) <= 4

    def _join_all_processes(self):
        for p in self._processes.values():
            p.join()

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

def run_hpo(hpo_algo: HpoBase, train_func: Callable):
    hpo_loop = HpoLoop(hpo_algo, train_func)
    best_config = hpo_loop.run()
    return best_config
