import queue
from typing import List

from ..utils import progressbar
from .easy_pipeline import SimplePipeline
from .stage import Stage
from .task import LoggedTask


class Pipeline(object):

    def __init__(self, stages: List[Stage]):
        self.stages = stages
        self.pipeline_items = [s.get_pipeline_item() for s in stages]
        self.pipeline = SimplePipeline(self.pipeline_items, 64)
        self.job_queue = self.pipeline.job_queue
        self.result_queue = self.pipeline.result_queue
        self.pipeline.start()

    def close(self):
        self.pipeline.stop()
        self.pipeline.close()

    def _wait(self, timeout):
        while True:
            try:
                task = self.result_queue.get(timeout=timeout)
                # TODO: filter out previous tasks
                yield task
                if task.meta.get('end', False):
                    return
            except queue.Empty:
                raise ChildProcessError(
                    'Wait time out after %d seconds.' % (timeout))

    def wait(self, timeout, show_progress, **progressbar_args):
        iterator = self._wait(timeout)
        if show_progress:
            if progressbar_args.get('total') is not None:
                progressbar_args['total'] += 1
            iterator = progressbar(iterator, **progressbar_args)
        yield from iterator

    def process(self, task, timeout=None, show_progress=True,
                **progressbar_args):
        self.job_queue.put(task)
        yield from self.wait(timeout, show_progress, total=task.value.limit,
                             **progressbar_args)
