# pylint: disable=no-member
# pylint: disable=unsubscriptable-object
import json
import os.path as osp
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit  # pylint: disable=no-name-in-module
from scipy.stats import linregress, norm

from .base import Monitor, ObjectType
from .interest import get_movement_heatmaps, get_region_mask

TRACK_DIR = osp.join(osp.dirname(__file__), 'tracks')
REGION_DIR = osp.join(osp.dirname(__file__), 'regions')
TrackItem = namedtuple('TrackItem', ['frame_id', 'obj_type', 'data'])
Event = namedtuple('Event', [
    'video_id', 'frame_id', 'movement_id', 'obj_type', 'confidence', 'track_id',
    'track'])


class MovementMonitor(Monitor):

    def __init__(
            self, video_name, fps, stride, video_id, camera_id, img_height,
            img_width, gaussian_std=0.3, min_length=0.3, speed_window=1,
            min_speed=10, type_thres=0.7, no_truck=False,
            distance_scale=5, distance_base_size=4, distance_slope_scale=2,
            proportion_thres_to_delta=0.5, proportion_scale=0.8,
            start_period=0.3, start_thres=0.5, start_proportion_factor=1.5,
            merge_detection_score=False, min_score=0.25,
            return_all_events=False):
        super().__init__(video_name, fps, stride)
        self.video_id = video_id
        self.camera_id = camera_id
        self.region = np.loadtxt(
            osp.join(REGION_DIR, 'cam_%d.txt' % (camera_id)), delimiter=',',
            dtype=np.int)
        with open(osp.join(TRACK_DIR, 'cam_%d.json' % (camera_id))) as f:
            data = json.load(f)
        self.movements = {int(shape['label']): np.array(shape['points'])
                          for shape in data['shapes']}
        assert len(self.movements) == max(self.movements.keys())
        self.img_height, self.img_width = img_height, img_width
        self.region_mask = get_region_mask(self.region, img_height, img_width)
        self.distance_heatmaps, self.proportion_heatmaps = \
            get_movement_heatmaps(self.movements, img_height, img_width)
        self.gaussian_std = gaussian_std
        if gaussian_std is not None:
            self.gaussian_std = gaussian_std * fps
        self.min_length = max(3, min_length * fps)
        self.speed_window = int(speed_window * fps // 2) * 2
        self.min_speed = min_speed / fps * self.speed_window
        self.type_thres = type_thres
        self.no_truck = no_truck
        self.distance_scale = distance_scale
        self.distance_base_size = distance_base_size
        self.distance_slope_scale = distance_slope_scale
        self.proportion_thres_to_delta = proportion_thres_to_delta
        self.proportion_factor = 1 / proportion_scale
        self.start_period = start_period * fps
        self.start_thres = start_thres
        self.start_proportion_factor = start_proportion_factor
        self.merge_detection_score = merge_detection_score
        self.min_score = min_score
        self.return_all_events = return_all_events
        self.tracks = defaultdict(list)
        self.obj_mask = {}
        self.last_frame_id = -1

    def forward_tracks(self, instances, image_id):
        self.last_frame_id = image_id
        boxes = instances.track_boxes.numpy()
        locations = ((boxes[:, :2] + boxes[:, 2:]) / 2)
        locations[:, 0] = np.clip(locations[:, 0], 0, self.img_width - 1)
        locations[:, 1] = np.clip(locations[:, 1], 0, self.img_height - 1)
        diagonals = np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1)
        boxes[:, ::2] = np.clip(boxes[:, ::2], 0, self.img_width)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, self.img_height)
        xy0s = np.floor(boxes[:, :2]).astype(np.int)
        xy1s = np.ceil(boxes[:, 2:]).astype(np.int)
        for obj_i in range(len(instances)):
            track_id = instances.track_ids[obj_i].item()
            if track_id == 0:
                continue
            obj_type = instances.pred_classes[obj_i].item()
            score = instances.scores[obj_i].item()
            diagonal = diagonals[obj_i]
            x, y = locations[obj_i]
            x_int = x.round().astype(int)
            mask_column = instances.pred_masks[obj_i, :, x_int]
            mask_bottom = np.where(mask_column.numpy())[0]
            if mask_bottom.shape[0] > 0:
                y = max(locations[obj_i, 1], mask_bottom.max())
            y_int = y.round().astype(int)
            if not self.region_mask[y_int, x_int]:
                x0, y0 = xy0s[obj_i]
                x1, y1 = xy1s[obj_i]
                box_size = (y1 - y0) * (x1 - x0)
                overlap_size = self.region_mask[y0:y1, x0:x1].sum()
                overlap_frac = overlap_size / box_size
                if overlap_frac == 0:
                    continue
                obj_mask = instances.pred_masks[obj_i].numpy()
                overlap_mask = np.logical_and(obj_mask, self.region_mask)
                overlap_frac = overlap_mask.sum() / obj_mask.sum()
                if overlap_frac == 0:
                    continue
            track_item = TrackItem(image_id, obj_type, (x, y, diagonal, score))
            self.tracks[track_id].append(track_item)
            self.obj_mask[track_id] = instances.pred_masks[obj_i].numpy()

    def get_track(self, track_items):
        init_frame_id = track_items[0].frame_id
        length = track_items[-1].frame_id - init_frame_id + 1
        if length < self.min_length:
            return None
        if len(track_items) == length:
            interpolated_track = np.stack([t.data for t in track_items])
        else:
            interpolated_track = np.empty((length, len(track_items[0].data)))
            interpolated_track[:, 0] = -1
            for t in track_items:
                interpolated_track[t.frame_id - init_frame_id] = t.data
            for frame_i, state in enumerate(interpolated_track):
                if state[0] >= 0:
                    continue
                for left in range(frame_i - 1, -1, -1):
                    if interpolated_track[left, 0] >= 0:
                        left_state = interpolated_track[left]
                        break
                for right in range(frame_i + 1, interpolated_track.shape[0]):
                    if interpolated_track[right, 0] >= 0:
                        right_state = interpolated_track[right]
                        break
                movement = right_state - left_state
                ratio = (frame_i - left) / (right - left)
                interpolated_track[frame_i] = left_state + ratio * movement
        if self.gaussian_std is not None:
            track = gaussian_filter1d(
                interpolated_track, self.gaussian_std, axis=0, mode='nearest')
        else:
            track = interpolated_track
        track = np.hstack([track, np.arange(
            init_frame_id, init_frame_id + length)[:, None]])
        speed_window = min(self.speed_window, track.shape[0] - 1)
        speed_window_half = speed_window // 2
        speed_window = speed_window_half * 2
        speed = np.linalg.norm(
            track[speed_window:, :2] - track[:-speed_window, :2], axis=1)
        speed_mask = np.zeros((track.shape[0]), dtype=np.bool)
        speed_mask[speed_window_half:-speed_window_half] = \
            speed >= self.min_speed
        speed_mask[:speed_window_half] = speed_mask[speed_window_half]
        speed_mask[-speed_window_half:] = speed_mask[-speed_window_half - 1]
        track = track[speed_mask]
        track_int = track[:, :2].round().astype(int)
        iou_mask = self.region_mask[track_int[:, 1], track_int[:, 0]]
        track = track[iou_mask]
        if track.shape[0] < self.min_length:
            return None
        return track

    def get_obj_type(self, track_items, track):
        active_frame_ids = set(track[:, -1].tolist())
        obj_types = [t.obj_type for t in track_items
                     if t.frame_id in active_frame_ids]
        type_counts = np.bincount(obj_types)
        obj_type = ObjectType.Car
        if not self.no_truck and \
                type_counts.shape[0] > 2:
            type_counts_sum = type_counts.sum()
            truck_score = type_counts[2] / type_counts_sum
            if truck_score > self.type_thres:
                obj_type = ObjectType.Truck
        return obj_type

    def get_movement_scores(self, track, obj_type, final=True):
        positions = track[:, :2].round().astype(int)
        diagonals = track[:, 2]
        detection_scores = track[:, 3]
        frame_ids = track[:, -1]
        distances = self.distance_heatmaps[:, positions[:, 1], positions[:, 0]]
        proportions = self.proportion_heatmaps[
            :, positions[:, 1], positions[:, 0]]
        distances = distances / diagonals[None]
        mean_distances = distances.mean(axis=1)
        x = np.linspace(0, 1, proportions.shape[1])
        distance_slopes = np.empty((len(self.movements)))
        proportion_slopes = np.empty((len(self.movements)))
        for movement_i in range(len(self.movements)):
            distance_slopes[movement_i] = linregress(
                x, distances[movement_i])[0]
            proportion_slopes[movement_i] = linregress(
                x, proportions[movement_i])[0]
        proportion_delta = proportions.max(axis=1) - proportions.min(axis=1)
        proportion_slopes = np.where(
            proportion_slopes >= self.proportion_thres_to_delta,
            proportion_delta, proportion_slopes)
        if obj_type == ObjectType.Truck:
            distance_base_scale = min(
                1, self.distance_base_size / mean_distances.shape[0])
            distance_base = np.sort(mean_distances)[
                :self.distance_base_size].sum() * distance_base_scale
            score_1 = 1 - (mean_distances / distance_base) ** 2
        else:
            score_1 = expit(4 - mean_distances * self.distance_scale)
        score_2 = self.proportion_factor * np.minimum(
            proportion_slopes, 1 / (proportion_slopes + 1e-8))
        if frame_ids[0] <= self.start_period and \
                score_2.max() <= self.start_thres:
            score_2 *= self.start_proportion_factor
        score_3 = norm.pdf(distance_slopes * self.distance_slope_scale) / 0.4
        scores = np.stack([score_1, score_2, score_3], axis=1)
        if final:
            scores = np.clip(scores, 0, 1).prod(axis=1)
            if self.merge_detection_score:
                scores = scores * detection_scores.mean()
        return scores

    def get_event(self, track_id):
        track_items = self.tracks.pop(track_id, None)
        if track_items is None:
            return None
        track = self.get_track(track_items)
        if track is None:
            return None
        obj_type = self.get_obj_type(track_items, track)
        frame_id = (track_items[-1][0] + 1) * self.stride
        movement_scores = self.get_movement_scores(track, obj_type)
        max_index = movement_scores.argmax()
        max_score = movement_scores[max_index]
        if max_score < self.min_score:
            if not self.return_all_events:
                return None
            movement_id = 0
        else:
            movement_id = max_index + 1
        event = Event(
            self.video_id, frame_id, movement_id, obj_type, max_score,
            track_id, track_items)
        return event

    def monit(self, frame):
        self.forward_tracks(frame.instances, frame.image_id)
        finished_tracks = frame.instances.finished_tracks.flatten(
            end_dim=-2)[:, -1].numpy()
        events = []
        for track_id in finished_tracks:
            if track_id == 0:
                break
            self.obj_mask.pop(track_id, None)
            event = self.get_event(track_id)
            if event is None:
                continue
            events.append(event)
        return events

    def finish(self):
        events = []
        for track_id in [*self.tracks.keys()]:
            obj_mask = self.obj_mask[track_id]
            overlap_mask = np.logical_and(obj_mask, self.region_mask)
            overlap_frac = overlap_mask.sum() / obj_mask.sum()
            if overlap_frac == 1:
                continue
            event = self.get_event(track_id)
            if event is None:
                continue
            events.append(event)
        return events


def evaluate(gt_events, pred_events, n_movements, n_frames, n_segments):
    segment_ends = np.arange(
        1, n_segments + 1)[:, None] * n_frames / n_segments
    segment_weights = np.arange(
        1, n_segments + 1) * 2 / (n_segments * (n_segments + 1))
    values = np.zeros((len(ObjectType), n_movements))
    weights = np.zeros_like(values)
    matched_count = 0
    details = []
    for obj_type in ObjectType:
        for movement_id in range(1, n_movements + 1):
            key = (obj_type, movement_id)
            gt_frame_ids = np.array(gt_events.get(key, []))
            pred_frame_ids = np.array(pred_events.get(key, []))
            n_vehicle = gt_frame_ids.shape[0]
            if n_vehicle == 0:
                continue
            matched_count += pred_frame_ids.shape[0]
            gt_counts = (gt_frame_ids[None] <= segment_ends).sum(axis=1)
            pred_counts = (pred_frame_ids[None] <= segment_ends).sum(axis=1)
            squared_error = (gt_counts - pred_counts) ** 2
            weighted_mse = (squared_error * segment_weights).sum()
            normalized_rmse = max(0, 1 - np.sqrt(weighted_mse) / n_vehicle)
            values[obj_type - 1, movement_id - 1] = normalized_rmse
            weights[obj_type - 1, movement_id - 1] = n_vehicle
            details.append((obj_type, movement_id, normalized_rmse,
                            n_vehicle, gt_counts, pred_counts))
    score = np.average(values, weights=weights)
    details = sorted(details, key=lambda x: x[-2][-1], reverse=True)
    gt_count = sum([len(e) for e in gt_events.values()])
    pred_count = sum([len(e) for e in pred_events.values()])
    details.append((-1, -1, -1, matched_count - pred_count,
                    gt_count, pred_count))
    details = pd.DataFrame(details, columns=[
        'obj_type', 'movement_id', 'normalized_rmse', 'n_vehicle',
        'gt_counts', 'pred_counts'])
    return score, details


def assign_events_to_frames(frames, events):
    events_lookup = {e.track_id: e for e in events}
    for frame in frames:
        instances = frame.instances
        event_ids = np.zeros((len(instances)), dtype=np.int)
        event_scores = np.zeros((len(instances)))
        event_counted = np.zeros((len(instances)), dtype=np.bool)
        event_types = np.zeros((len(instances)), dtype=np.int)
        for obj_i in range(len(instances)):
            track_id = instances.track_ids[obj_i].item()
            event = events_lookup.get(track_id)
            if event is None:
                continue
            event_ids[obj_i] = event.movement_id
            event_scores[obj_i] = event.confidence
            event_counted[obj_i] = frame.image_id + 1 >= event.frame_id
            event_types[obj_i] = event.obj_type
        instances.event_ids = event_ids
        instances.event_scores = event_scores
        instances.event_counted = event_counted
        instances.event_types = event_types


def count_events_by_frame(events, n_frames, n_events):
    event_counts = np.zeros((n_frames, n_events, 2), dtype=np.int)
    for event in events:
        event_counts[event.frame_id - 1:, event.movement_id - 1,
                     event.obj_type - 1] += 1
    return event_counts
