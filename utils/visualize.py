import argparse
import multiprocessing as mp
import os
import os.path as osp
import time

import cv2

from ..detector import Detector
from ..loader import MoviePy as Loader
from ..monitor import Monitor
from ..monitor.movement import assign_events_to_frames, count_events_by_frame
from ..tracker import Tracker
from ..utils import progressbar
from ..visualizer import Visualizer
from .run import get_jobs


def video_worker(video_job, stride, dataset_dir, output_dir,
                 gpu_id, worker_id):
    loader = Loader(video_job.video_name, dataset_dir)
    real_fps = loader.fps / stride
    detector = Detector(gpu_id)
    tracker = Tracker(video_job.video_name, real_fps,
                      **video_job.kwargs.get('tracker_args', {}))
    monitor = Monitor(
        video_job.video_name, real_fps, stride, video_job.video_id,
        video_job.camera_id, loader.video.size[1], loader.video.size[0])
    visualizer = Visualizer()

    loader_iter = loader.read_iter(
        limit=video_job.n_frames, stride=stride, start=video_job.start_frame)
    frames, events = [], []
    for _ in progressbar(
            range(video_job.n_frames // stride),
            'Processing (%d) %s' % (worker_id, video_job.video_name),
            position=worker_id + 1, leave=False):
        images, image_ids = next(loader_iter)
        frame = detector.detect(images, image_ids)[0]
        frame = tracker.track(frame)
        event = monitor.monit(frame)
        frames.append(frame)
        events.extend(event)
    event = monitor.finish()
    events.extend(event)

    assign_events_to_frames(frames, events)
    event_counts = count_events_by_frame(
        events, len(frames), len(monitor.movements))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    filename = osp.join(output_dir, '%s_%d-%d.mp4' % (
        osp.splitext(video_job.video_name)[0], video_job.start_frame,
        video_job.start_frame + image_ids[-1] + 1))
    writer = cv2.VideoWriter(
        filename, fourcc, int(loader.fps), tuple(loader.video.size))
    try:
        for frame_i in progressbar(
                range(len(frames)),
                'Visualizing (%d) %s' % (worker_id, video_job.video_name),
                position=worker_id + 1, leave=False):
            visual_image = visualizer.draw_scene(
                frames[frame_i], monitor, event_counts[frame_i])
            frames[frame_i] = None
            writer.write(visual_image[:, :, ::-1])
    finally:
        writer.release()
    return filename


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    n_detector = args.n_gpu * args.n_detector_per_gpu
    jobs = get_jobs(args.video_list_file)
    processes = [None] * n_detector
    for i in range(1, args.n_detector_per_gpu):
        for worker_id in range(i * args.n_gpu, (i + 1) * args.n_gpu):
            process = mp.Process(target=lambda: time.sleep(120))
            process.start()
            processes[worker_id] = process
    for job in progressbar(jobs, 'Jobs'):
        while True:
            for worker_id, process in enumerate(processes):
                if process is not None and not process.is_alive():
                    process.join()
                    processes[worker_id] = None
            for worker_id, process in enumerate(processes):
                if process is None:
                    break
            if process is None:
                break
            time.sleep(1)
        run_args = (job, args.stride, args.dataset_dir, args.output_dir,
                    worker_id % args.n_gpu, worker_id)
        process = mp.Process(target=video_worker, args=run_args)
        process.start()
        processes[worker_id] = process
    for worker_id, process in enumerate(processes):
        if process is not None and not process.is_alive():
            process.join()
            processes[worker_id] = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        '%s.%s' % (__package__, osp.splitext(osp.basename(__file__))[0]))
    parser.add_argument(
        'dataset_dir', help='Path to dataset directory')
    parser.add_argument(
        'video_list_file', help='Path to video list file')
    parser.add_argument(
        'output_dir', help='Path to output directory')
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--n_gpu', default=4, type=int)
    parser.add_argument('--n_detector_per_gpu', default=1, type=int)
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(parse_args())
