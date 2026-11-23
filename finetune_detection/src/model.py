"""Torchvision Faster R-CNN with a two-class detection head."""

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ARCHITECTURE = "fasterrcnn_resnet50_fpn_v2"
LABELS = {0: "__background__", 1: "skin_lesion"}


def build_model(config, pretrained=True):
    if config["model"] != ARCHITECTURE:
        raise ValueError(f"Unsupported architecture: {config['model']}")
    layers = config["trainable_backbone_layers"]
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None,
        trainable_backbone_layers=layers if pretrained else None,
        min_size=config["min_size"],
        max_size=config["max_size"],
        box_score_thresh=0.0,
        box_detections_per_img=100,
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, num_classes=len(LABELS))
    # Torchvision ignores trainable_backbone_layers when loading no base weights.
    stages = ["layer4", "layer3", "layer2", "layer1", "conv1"][:layers]
    if layers == 5:
        stages.append("bn1")
    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad_(name.split(".")[0] in stages)
    return model


def load_inference(path, device="cpu"):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact["format"] != "isic-fasterrcnn-inference-v1" or artifact["labels"] != LABELS:
        raise ValueError("Unsupported inference artifact")
    model = build_model(artifact["model_config"], pretrained=False)
    model.load_state_dict(artifact["model"], strict=True)
    return model.to(device).eval(), artifact
