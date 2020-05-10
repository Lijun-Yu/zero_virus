from functools import partial

from .moviepy import MoviePy
from .stage import LoaderStage as LoaderStage_base

__all__ = ['LoaderStage']

loader_classes = {'mp4': MoviePy}
LoaderStage = partial(LoaderStage_base, loader_classes)
