import numpy as np
import torch
from moviepy.editor import VideoFileClip

from .base import Loader


class MoviePy(Loader):

    def __init__(self, video_path, parent_dir=''):
        super().__init__(video_path, parent_dir)
        self.video = VideoFileClip(self.path, audio=False)
        self.fps = self.video.fps

    def read_iter(self, batch_size=1, limit=None, stride=1, start=0):
        # start and limit should be in original frame indices
        images, image_ids = [], []
        positions = np.arange(
            0, self.video.duration * self.fps, stride / self.fps)
        start = start // stride
        end = start + limit // stride if limit is not None else None
        for pos in positions[:start]:
            self.video.get_frame(pos)
        for image_id, pos in enumerate(positions[start:end]):
            image = self.video.get_frame(pos)
            image = np.ascontiguousarray(image[:, :, ::-1])
            image = torch.as_tensor(image)
            images.append(image)
            image_ids.append(image_id)
            if len(images) == batch_size:
                yield images, image_ids
                images, image_ids = [], []
        if len(images) > 0:
            yield images, image_ids
