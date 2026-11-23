"""COCO bounding-box AP and fixed operating-point metrics."""

import contextlib
import io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou

from .annotations import validate_box


def xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


class DetectionMetrics:
    def __init__(self, threshold):
        self.threshold = threshold
        self.images = {}
        self.annotations = []
        self.predictions = []
        self.true_positives = 0
        self.false_positives = 0
        self.matched_iou = 0.0

    def update(self, prediction, target, width, height):
        image_id = int(target["image_id"])
        if image_id in self.images:
            raise ValueError(f"Duplicate evaluation image ID {image_id}")
        self.images[image_id] = {"id": image_id, "width": width, "height": height}
        truth = target["boxes"]
        if len(truth) != len(target["labels"]) or not (target["labels"] == 1).all():
            raise ValueError(f"Invalid evaluation targets for image {image_id}")
        for box in truth.tolist():
            self.annotations.append({
                "image_id": image_id, "id": len(self.annotations) + 1, "category_id": 1,
                "bbox": xywh(box), "area": validate_box(box, width, height), "iscrowd": 0,
            })
        boxes, scores = prediction["boxes"], prediction["scores"]
        if (not len(boxes) == len(scores) == len(prediction["labels"])
                or not ((scores >= 0) & (scores <= 1)).all() or not (prediction["labels"] == 1).all()):
            raise ValueError(f"Invalid detections for image {image_id}")
        ranked = scores.argsort(descending=True, stable=True)[:100]
        boxes, scores = boxes[ranked], scores[ranked]
        for box, score in zip(boxes.tolist(), scores.tolist()):
            validate_box(box, width, height)
            self.predictions.append({"image_id": image_id, "category_id": 1, "bbox": xywh(box), "score": score})
        ious = box_iou(boxes[scores >= self.threshold], truth)
        for overlaps in ious:
            if overlaps.numel() and float(overlaps.max()) >= 0.5:
                index = int(overlaps.argmax())
                self.true_positives += 1
                self.matched_iou += float(overlaps[index])
                ious[:, index] = -1
            else:
                self.false_positives += 1

    def compute(self):
        if not self.images or not self.annotations:
            raise ValueError("Evaluation requires images and at least one lesion annotation")
        map_50_95, map_50 = 0.0, 0.0
        if self.predictions:
            ground_truth = COCO()
            ground_truth.dataset = {
                "info": {}, "images": list(self.images.values()), "annotations": self.annotations,
                "categories": [{"id": 1, "name": "skin_lesion"}],
            }
            with contextlib.redirect_stdout(io.StringIO()):
                ground_truth.createIndex()
                evaluation = COCOeval(ground_truth, ground_truth.loadRes(self.predictions), iouType="bbox")
                evaluation.evaluate()
                evaluation.accumulate()
                evaluation.summarize()
            map_50_95, map_50 = map(float, evaluation.stats[:2])
        return {
            "map_50": map_50, "map_50_95": map_50_95,
            "recall_50": self.true_positives / len(self.annotations),
            "mean_matched_iou": self.matched_iou / self.true_positives if self.true_positives else None,
            "fp_per_image": self.false_positives / len(self.images),
        }
