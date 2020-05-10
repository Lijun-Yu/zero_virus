import matplotlib as mpl
import numpy as np
import torch
from detectron2.utils.visualizer import Visualizer as dt_visualizer
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle

from ..detector.base import ObjectType
from .color import ColorManager


class Visualizer(object):

    def __init__(self, box_3d_color_sync=False):
        self.color_manager = ColorManager()
        self.box_3d_color_sync = box_3d_color_sync

    def plt_imshow(self, image, figsize=(16, 9), dpi=120, axis='off',
                   show=True):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.axis(axis)
        plt.imshow(image)
        if show:
            plt.show()
            plt.close(fig)

    def draw_scene(self, frame, monitor=None, event_count=None):
        image_rgb = frame.image.numpy()[:, :, ::-1]
        visualizer = dt_visualizer(image_rgb, None)
        visualizer.draw_text('Frame %d' % (frame.image_id), (10, 10),
                             font_size=15, horizontal_alignment='left')
        instances = frame.instances.to('cpu')
        obj_colors = self._draw_instances(visualizer, instances, image_rgb)
        if instances.has('boxes_3d'):
            for obj_i in range(len(instances)):
                angle = instances.angles[obj_i]
                if angle < 0:
                    continue
                box_3d = instances.boxes_3d[obj_i]
                color = obj_colors[obj_i] if self.box_3d_color_sync else None
                self._draw_box_3d(visualizer, box_3d, color)
        if instances.has('boxes_3d_gt'):
            for obj_i in range(len(instances)):
                box_3d = instances.boxes_3d_gt[obj_i]
                if box_3d[0, 0, 0] < 0:
                    continue
                color = obj_colors[obj_i] if self.box_3d_color_sync else None
                self._draw_box_3d(
                    visualizer, box_3d, color, linestyle='dotted')
        if monitor is not None:
            movement_colors = self._draw_region_movements(
                visualizer, image_rgb, monitor.region, monitor.movements,
                event_count)
            self._draw_events(visualizer, instances, movement_colors)
        output = visualizer.get_output()
        visual_image = output.get_image()
        plt.close(output.fig)
        return visual_image

    def _draw_instances(self, visualizer, instances, image_rgb):
        labels, colors, masks = [], [], []
        for obj_i in range(len(instances)):
            obj_type = ObjectType(instances.pred_classes[obj_i].item())
            obj_id = obj_i
            if instances.has('track_ids'):
                obj_id = instances.track_ids[obj_i].item()
            obj_type = obj_type.name
            score = instances.scores[obj_i] * 100
            label = '%s-%s %.0f%%' % (obj_type, obj_id, score)
            if instances.has('event_ids'):
                event_id = instances.event_ids[obj_i]
                if event_id != 0:
                    event_type = ObjectType(instances.event_types[obj_i])
                    event_score = instances.event_scores[obj_i] * 100
                    label += ' E-%s-%d %.0f%%' % (
                        event_type.name, event_id, event_score)
            labels.append(label)
            x0, y0, x1, y1 = instances.pred_boxes.tensor[obj_i].type(torch.int)
            roi = image_rgb[y0:y1, x0:x1]
            color = self.color_manager.get_color(obj_id, roi)
            colors.append(color)
            mask = [np.array([0, 0])]
            if instances.has('pred_masks'):
                mask = instances.pred_masks[obj_i].numpy()
            if instances.has('contours'):
                contour = instances.contours[obj_i]
                if contour is not None:
                    contour = contour[contour[:, 0] >= 0]
                    mask = [contour.cpu().numpy()]
            masks.append(mask)
        visualizer.overlay_instances(
            masks=masks, boxes=instances.pred_boxes, labels=labels,
            assigned_colors=colors)
        return colors

    def _linewidth(self, visualizer):
        linewidth = max(
            visualizer._default_font_size / 4, 1) * visualizer.output.scale
        return linewidth

    def _draw_box_3d(self, visualizer, box_3d, color=None, alpha=0.5,
                     linestyle='dashdot'):

        if color is None:
            colors = ['red'] * 4 + ['green'] * 4 + ['blue'] * 4
        else:
            colors = [color] * 16
        bottom = box_3d[:, :, 1].mean(axis=1).argmax()
        xs_list, ys_list = [], []
        for layer_i in [bottom, 1 - bottom]:
            for point_i in range(4):
                xs_list.append([box_3d[layer_i, point_i, 0],
                                box_3d[layer_i, (point_i + 1) % 4, 0]])
                ys_list.append([box_3d[layer_i, point_i, 1],
                                box_3d[layer_i, (point_i + 1) % 4, 1]])
        for point_i in range(4):
            xs_list.append(
                [box_3d[0, point_i, 0], box_3d[1, 3 - point_i, 0]])
            ys_list.append(
                [box_3d[0, point_i, 1], box_3d[1, 3 - point_i, 1]])
        ax = visualizer.output.ax
        for xs, ys, color in zip(xs_list, ys_list, colors):
            ax.add_line(mpl.lines.Line2D(
                xs, ys, linewidth=self._linewidth(visualizer),
                color=color, alpha=alpha, linestyle=linestyle))

    def _draw_region_movements(self, visualizer, image_rgb, region, movements,
                               event_count=None, arrow_size=20):
        ax = visualizer.output.ax
        x0, y0 = region.min(axis=0).astype(int)
        x1, y1 = region.max(axis=0).astype(int)
        roi = image_rgb[y0:y1, x0:x1]
        color = self.color_manager.get_color('region', roi)
        roi = Polygon(region, fill=False, color=color,
                      linewidth=self._linewidth(visualizer))
        ax.add_patch(roi)
        colors = {}
        movements = sorted(movements.items(), key=lambda x: int(x[0]))
        if event_count is not None:
            event_count = event_count.astype(str)
        for label, movement in movements:
            x0, y0 = movement.min(axis=0).astype(int)
            x1, y1 = movement.max(axis=0).astype(int)
            roi = image_rgb[y0:y1, x0:x1]
            color = self.color_manager.get_color(('movement', label), roi)
            colors[label] = color
            if event_count is not None:
                label = '%s: %s' % (label, ', '.join(event_count[label - 1]))
            line = Polygon(
                movement, closed=False, fill=False, edgecolor=color,
                label=label, linewidth=self._linewidth(visualizer))
            ax.add_patch(line)
            dxy = movement[-1] - movement[-2]
            dxy = dxy / np.linalg.norm(dxy) * arrow_size
            origin = movement[-1] - dxy
            arrow = FancyArrow(
                *origin, *dxy, edgecolor=color, facecolor=color,
                width=2 * self._linewidth(visualizer),
                head_length=arrow_size, head_width=arrow_size)
            ax.add_patch(arrow)
        ax.legend(loc=0)
        return colors

    def _draw_events(self, visualizer, instances, colors):
        ax = visualizer.output.ax
        for obj_i in range(len(instances)):
            event_id = instances.event_ids[obj_i].item()
            if event_id == 0:
                continue
            box = instances.pred_boxes.tensor[obj_i]
            center = (box[:2] + box[2:]) / 2
            size = (box[2:] - box[:2]).max() / 10
            color = colors[event_id]
            counted = instances.event_counted[obj_i]
            if not counted:
                patch = Circle(center, size, color=color)
            else:
                patch = Rectangle(center, size, size, color=color)
            ax.add_patch(patch)
