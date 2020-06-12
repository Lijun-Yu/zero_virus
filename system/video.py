import json
import os
import sys
import time
from collections import namedtuple
from itertools import chain

from ..detector import DetectorStage
from ..detector.base import Frame
from ..loader import LoaderStage
from ..loader.stage import LoaderTask
from ..monitor import MinotorStage
from ..monitor.stage import MonitorResult
from ..pipeline import LoggedTask, Pipeline
from ..tracker import TrackerStage
from ..utils import progressbar
from .base import System
from .output import get_output

VideoJob = namedtuple('Job', [
    'video_name', 'video_id', 'camera_id', 'n_frames', 'start_frame',
    'kwargs'], defaults=[None, 0, {}])


class VideoSystem(System):

    def __init__(self, video_dir, n_gpu=128, batch_size_per_gpu=4,
                 n_detector_per_gpu=1, gpu_per_replica=0.5):
        self.video_dir = video_dir
        self.batch_size = batch_size_per_gpu
        self.stride = 1
        super(VideoSystem, self).__init__(n_gpu, gpu_per_replica)
        for pipeline_i, gpu in enumerate(self.gpus):
            stages = [LoaderStage(), DetectorStage([gpu], n_detector_per_gpu),
                      TrackerStage(), MinotorStage()]
            self.pipelines[pipeline_i] = Pipeline(stages)
        self.start()

    def get_result(self, results_gen):
        events = []
        for task in results_gen:
            if hasattr(task.value, 'events') and task.value.events is not None:
                events.extend(task.value.events)
        return events

    def process(self, jobs):
        for job in jobs:
            loader_task = LoaderTask(
                job.video_name, self.video_dir, job.video_id, job.camera_id,
                self.batch_size, job.n_frames, self.stride, job.start_frame,
                job.kwargs)
            task = LoggedTask(
                loader_task, meta={}, start_time=time.time())
            self.jobs.put((job.video_name, task))
        for _ in range(self.replica):
            self.jobs.put(None)
        for job_i in progressbar(
                range(len(jobs)), 'Jobs', position=self.replica):
            while True:
                time.sleep(0.1)
                if len(self.results) > job_i:
                    break

    def get_output(self):
        return get_output(chain(*self.results))
