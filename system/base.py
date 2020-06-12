import queue
from threading import Lock, Thread
from typing import List, Union
from itertools import cycle

import GPUtil

from ..pipeline import LoggedTask, Pipeline, Stage
from ..utils import get_logger, progressbar


class System(object):

    def __init__(self, n_gpu: int, gpu_per_replica: Union[int, float] = 1):
        self.logger = get_logger(self.__class__.__name__)
        available_gpus = GPUtil.getAvailable(
            limit=n_gpu, maxLoad=0.2, maxMemory=0.2)
        self.replica = int(max(1, len(available_gpus) / gpu_per_replica))
        if isinstance(gpu_per_replica, float):
            self.gpus = available_gpus * int(1 / gpu_per_replica)
        else:
            self.gpus = available_gpus[:self.replica * gpu_per_replica]
        if len(self.gpus) == 0:
            self.logger.warn(
                'No gpus available, running %d replica on cpu', self.replica)
        else:
            self.logger.info(
                'Running %d replicas on %d gpus (gpu id: %s)',
                self.replica, len(available_gpus), available_gpus)
        self.pipelines = [None for _ in range(self.replica)]
        self.threads = [Thread(target=self.single_pipeline, args=(i,))
                        for i in range(self.replica)]
        self.jobs = queue.Queue()
        self.results = []
        self.result_lock = Lock()

    def get_result(self, results_gen):
        return [*results_gen]

    def single_pipeline(self, pipeline_id):
        pipeline = self.pipelines[pipeline_id]
        while True:
            job = self.jobs.get()
            if job is None:
                return
            name, job = job
            try:
                results_gen = pipeline.process(
                    job, desc=' R%d(%s)' % (pipeline_id, name),
                    position=pipeline_id)
                results = self.get_result(results_gen)
            except Exception as e:
                print(e, flush=True)
            with self.result_lock:
                self.results.append(results)

    def start(self):
        for thread in self.threads:
            thread.start()

    def finish(self):
        for pipeline in self.pipelines:
            pipeline.close()

    def __repr__(self):
        return '%s.%s(%d)' % (
            self.__module__, self.__class__.__name__, self.replica)
