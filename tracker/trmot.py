from collections import defaultdict

import numpy as np
import torch

from ..utils import pack_tensor
from .base import Frame, ObjectType, Tracker
from .tr_mot.basetrack import TrackState
from .tr_mot.multitracker import JDETracker, STrack

TRACK_MAPPINGS = {ObjectType.Car: ObjectType.Car,
                  ObjectType.Truck: ObjectType.Car}


class TRMOT(Tracker):

    def __init__(self, video_name, fps, max_age=2,
                 min_iou=0.2, feature_thres=0.7, feature_buffer_size=1):
        super().__init__(video_name, fps)
        STrack.reset_id()
        self.trackers = {}
        for obj_type in set(TRACK_MAPPINGS.values()):
            self.trackers[obj_type] = JDETracker(
                int(max_age * fps), feature_thres, 1 - min_iou, 1 - min_iou / 2)
        self.feature_buffer_size = int(feature_buffer_size * fps)
        self.active_tracks = set()

    def group_instances(self, instances):
        grouped_instances = defaultdict(list)
        for obj_i in range(len(instances)):
            obj_type = ObjectType(instances.pred_classes[obj_i].item())
            obj_type = TRACK_MAPPINGS[obj_type]
            bbox = instances.pred_boxes.tensor[obj_i].numpy()
            tlwh = STrack.tlbr_to_tlwh(bbox)
            score = instances.scores[obj_i].item()
            feature = instances.roi_features[obj_i].numpy().copy()
            detection = STrack(
                tlwh, score, feature, obj_i, self.feature_buffer_size)
            grouped_instances[obj_type].append(detection)
        return grouped_instances

    def get_tracked_instances(self, instances):
        track_ids = torch.zeros((len(instances)), dtype=torch.int)
        states = torch.zeros((len(instances)), dtype=torch.int)
        track_boxes = torch.zeros((len(instances), 4))
        image_speeds = torch.zeros((len(instances), 2))
        for tracker in self.trackers.values():
            for track in tracker.tracked_stracks:
                self.active_tracks.add(track.track_id)
                obj_i = track.obj_index
                track_ids[obj_i] = track.track_id
                states[obj_i] = track.state
                track_boxes[obj_i] = torch.as_tensor(
                    track.tlbr, dtype=torch.float)
                speed = torch.as_tensor([
                    track.mean[4], track.mean[5] + track.mean[7] / 2])
                image_speeds[obj_i] = speed * self.fps
        instances.track_ids = track_ids
        instances.track_states = states
        instances.track_boxes = track_boxes
        instances.image_speeds = image_speeds
        if len(instances) > 0:
            ongoing_track_ids = set()
            for tracker in self.trackers.values():
                ongoing_track_ids.update([
                    t.track_id for t in tracker.tracked_stracks])
                ongoing_track_ids.update([
                    t.track_id for t in tracker.lost_stracks])
            finished_track_ids = self.active_tracks - ongoing_track_ids
            self.active_tracks = self.active_tracks - finished_track_ids
            instances.finished_tracks = pack_tensor(
                torch.as_tensor([*finished_track_ids]).unsqueeze(1),
                len(instances))
        else:
            instances.finished_tracks = torch.zeros((len(instances), 0, 1))
        return instances

    def track(self, frame):
        grouped_instances = self.group_instances(frame.instances)
        for obj_type, tracker in self.trackers.items():
            tracker.update(grouped_instances[obj_type])
        instances = self.get_tracked_instances(frame.instances)
        return Frame(frame.image_id, frame.image, instances)
