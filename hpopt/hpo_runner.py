import os
from functools import partial
from multiprocessing import Process, Queue
from typing import Any, Dict, Union


class HpoLoop:
    def __init__(self, hpo_algo, train_func):
        self._hpo_algo = hpo_algo
        self._train_func = train_func
        self._processes: Dict[int, Process] = {}
        self._report_queue = Queue()

    def run(self):
        while not self._hpo_algo.is_done():
            if self._have_resource_to_start_new_process():
                trial = self._hpo_algo.get_next_sample()
                if trial is not None:
                    self._start_trial_process(trial)

            self._remove_finished_process()
            self._get_reports_and_stop_trial_if_necessary()

        self._get_reports_and_stop_trial_if_necessary()

        return self._hpo_algo.get_best_config()

    def _start_trial_process(self, trial):
        process = Process(
            target=_run_hpo_trial,
            args=(
                trial.configuration,
                self._report_queue,
                trial.id,
                self._train_func
            )
        )
        self._processes[process.pid] = process
        process.start()

    def _remove_finished_process(self):
        processes_to_remove = []
        for pid, process in self._processes.items():
            if not process.is_alive():
                process.join()
                processes_to_remove.append(pid)

        for pid in processes_to_remove:
            del self._processes[pid]

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

def _run_hpo_trial(
    hp_config: Dict[str, Any],
    report_queue: Queue,
    trial_id: Any,
    train_func
):
    train_func(
        hp_config,
        partial(_report_score, report_queue=report_queue, trial_id=trial_id)
    )

def _report_score(
    score: Union[int, float],
    progress: Union[int, float],
    report_queue: Queue,
    trial_id: Any
):
    report_queue.put_nowait(
        {
            "score" : score,
            "progress" : progress,
            "trial_id" : trial_id,
            "pid" : os.getpid()
        }
    )

def run_hpo_loop(hpo_algo, train_func, ipc_pipe):
    hpo_loop = HpoLoop(hpo_algo, train_func)
    best_trial = hpo_loop.run()
    ipc_pipe.send(best_trial.configuration)
    ipc_pipe.close()
