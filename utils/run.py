import argparse
import json
import logging
import os
import os.path as osp
import sys

import pandas as pd

from ..pipeline import task
from ..system import VideoJob, VideoSystem
from . import log

logger = log.get_logger(__name__)


TRACKING_GROUPS = {**{cid: 0.5 for cid in [8, 14, 15, 16, 17, 18, 19, 20]},
                   **{cid: 0.1 for cid in [3]}}


def get_jobs(dataset_dir):
    video_list_file = osp.join(dataset_dir, 'list_video_id.txt')
    video_stats_file = osp.join(dataset_dir, 'track1_vid_stats.txt')
    video_list = pd.read_csv(
        video_list_file, sep=' ', index_col=0, header=None)
    video_stats = pd.read_csv(video_stats_file, sep='\t')
    frame_nums = {row['vid_name']: row['frame_num']
                  for _, row in video_stats.iterrows()}
    jobs = []
    for video_id, video in video_list.iterrows():
        video_name = video[1]
        camera_id = int(osp.splitext(video_name)[0].split('_')[1])
        n_frames = frame_nums[video_name]
        kwargs = {}
        if camera_id in TRACKING_GROUPS:
            kwargs['tracker_args'] = {'min_iou': TRACKING_GROUPS[camera_id]}
        jobs.append(VideoJob(
            video_name, video_id, camera_id, n_frames, 0, kwargs))
    jobs = sorted(jobs, key=lambda j: j.n_frames, reverse=True)
    return jobs


def main(args):
    logger.info('Running with args: %s', args)
    os.makedirs(osp.dirname(args.system_output), exist_ok=True)
    jobs = get_jobs(args.dataset_dir)
    system = VideoSystem(args.dataset_dir)
    logger.info('Running %d jobs', len(jobs))
    try:
        system.process(jobs)
    finally:
        output = system.get_output()
        output.to_csv(args.system_output, sep=' ', header=None, index=False)
        system.finish()
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        '%s.%s' % (__package__, osp.splitext(osp.basename(__file__))[0]))
    parser.add_argument(
        'dataset_dir', help='Path to dataset directory')
    parser.add_argument(
        'system_output', help='Path to output (output.json)')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    main(parse_args())
