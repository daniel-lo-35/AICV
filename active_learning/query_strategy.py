import torch
import os, cv2, argparse, json, math, random
import numpy as np

import detectron2
from detectron2.engine import DefaultPredictor

from detectron2.data import MetadataCatalog
from typing import List
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer 
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from termcolor import colored
from tabulate import tabulate
from datetime import datetime
import sys

import pycocotools.mask
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation.coco_evaluation import instances_to_coco_json


__all__ = ['StrategyBase', 'RandomStrategy', 'UncertaintyStrategy', 'RoIMatchingStrategy']


def _forward_boxes(model, data, return_features=False):
    model.eval()
    with torch.no_grad():
        images = model.preprocess_image(data)
        # features = model.backbone(images.tensor)
        features, _ = model.backbone(images.tensor)
        if model.proposal_generator is not None:
            proposals, _ = model.proposal_generator(images, features, None)
        else:
            assert "proposals" in data[0]
            proposals = [x["proposals"].to(model.device) for x in data]
        box_in_features = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(box_in_features, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)
        predictions = model.roi_heads.box_predictor(box_features)
        scores = model.roi_heads.box_predictor.predict_probs(predictions, proposals)
        pred_boxes, indices = model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_boxes[0].scores = scores[0][indices][:, :-1]
    if return_features:
        return pred_boxes[0], list(features.values())
    else:
        return pred_boxes[0]

class StrategyBase:
    """
    Base class for query strategy.
    """
    def __init__(self):
        pass
    
    @classmethod
    def sample(self, model, pool_name, k):
        """
        Sample k instances from data.
        Args:
            model (torch.nn.Module): a model for querying samples.
            pool_name (str): name of pool set.
            k (int): an interger number for sampling.
        Returns:
            image_id of sampled instances.
        """
        raise NotImplementedError
    
    def __str__(self):
        return f"{type(self).__name__}({', '.join(['{}={}'.format(attr, value) for attr, value in self.__dict__.items()])})"

class RandomStrategy(StrategyBase):
    """
    Random sampling query strategy. Selects instances randomly.
    Args:
        seed (int): random seed for sampling.
    """
    def __init__(self, seed=None):
        self.seed = seed
    
    def sample(self, cfg, pool_name, k, model=None, verbose=True):
        if model is None:
            model = build_model(cfg)
        random.seed(self.seed)
        loader = build_detection_test_loader(cfg, pool_name)
        ids = [data[0]["image_id"] for data in loader]
        selected = random.sample(ids, k) if len(ids) > k else ids
        if verbose:
            headers = ['Selected files', 'Scores']
            contents = [[s, None] for s in selected]
            ctable(headers, contents, title="RandomStrategy.Selected", color='yellow')
        return selected

class UncertaintyStrategy(StrategyBase):
    """
    Uncertainty sampling query strategy. Selects instances with the least confidence for labeling.
    Args:
        measurement (str): measurement method for uncertainty estimation which must be 
            in list of ['uncertainty', 'margin', 'entropy']
        aggregation (str): a aggregation method to aggregate measurements of detections which must be
            in list of ['sum', 'mean', 'max']
    """
    def __init__(self, measurement='uncertainty', aggregation='mean'):
        assert measurement in ['uncertainty', 'margin', 'entropy'], "measurement method must be in list of ['uncertainty', 'margin', 'entropy']"
        assert aggregation in ['sum', 'mean', 'max'], "aggregation function must be in list of ['sum', 'mean', 'max']"
        self.measurement = measurement
        self.aggregation = aggregation
    
    def get_uncertainty(self, scores):
        if self.measurement == "uncertainty":
            metrics = 1 - scores.max(-1)[0]
        elif self.measurement == "margin":
            metrics = torch.diff(torch.topk(scores, 2, dim=-1)[0], dim=-1).view(-1)
        elif self.measurement == "entropy":
            metrics = - torch.sum(scores * torch.log(scores), -1)
        
        if self.aggregation == "sum":
            metrics = torch.sum(metrics)
        elif self.aggregation == "mean":
            metrics = torch.mean(metrics)
        elif self.aggregation == "max":
            metrics = torch.max(metrics)
        return metrics
    
    def sample(self, cfg, pool_name, k, model=None, verbose=True, skip=1):
        if model is None:
            model = build_model(cfg)
        loader = build_detection_test_loader(cfg, pool_name)
        metrics_dict = {}
        for iter, data in enumerate(loader):
            if iter % skip == 0:
                pred_boxes = _forward_boxes(model, data)
                metrics = self.get_uncertainty(pred_boxes.scores)
                metrics_dict[data[0]["image_id"]] = float(metrics)
        
        rank = sorted(metrics_dict.items(), key=lambda x: x[1], reverse=True)    # Active Learning
        # rank = sorted(metrics_dict.items(), key=lambda x: x[1], reverse=False)    # Pseudo Labeling
        
        if verbose:
            headers = ['Selected files', 'Scores']
            contents = rank[:k]
            ctable(headers, contents, title="UncertaintyStrategy.Selected", color='yellow')
        return [r[0] for r in rank[:k]]

class RoIMatchingStrategy(StrategyBase):
    """
    Uncertainty estimation using MC dropout and RoI matching.
    Args:
        T (int): times of forward passes.
    """
    def __init__(self, T: int=3, measurement='vote', aggregation='mean'):
        assert T > 1, "T must greater than 1"
        assert measurement in ['vote', 'consensus', 'kld'], "measurement method must be in list of ['vote', 'consensus', 'kld']"
        assert aggregation in ['sum', 'mean', 'max'], "aggregation function must be in list of ['sum', 'mean', 'max']"
        self.T = T
        self.measurement = measurement
        self.aggregation = self.get_aggregation(aggregation)
    
    def get_iou(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def get_aggregation(self, aggregation):
        if aggregation == "sum":
            return np.sum
        elif aggregation == "mean":
            return np.mean
        elif aggregation == "max":
            return np.max
    
    def get_uncertainty(self, scores, num_classes):
        uncertainty = 0.0
        if self.measurement == "vote":
            labels = np.argmax(scores, axis=-1).tolist() + [-1] * (self.T-len(scores))
            for c in range(-1, num_classes):
                votes = labels.count(c)
                uncertainty += - votes/self.T * np.log(votes/self.T + 1e-10)
        elif self.measurement == "consensus":
            if len(scores) != self.T: return None
            consensus_scores = np.mean(scores, axis=0)
            uncertainty = - np.sum(consensus_scores * np.log(consensus_scores + 1e-10))
            
        elif self.measurement == "kld":
            if len(scores) != self.T: return None
            consensus_scores = np.mean(scores, axis=0)
            for score in scores:
                uncertainty += np.sum(score * np.log(score / consensus_scores))
            uncertainty /= self.T
        return uncertainty
    
    def sample(self, cfg, pool_name, k, model=None, verbose=True):
        # cfg.MODEL.RESNETS.MCDROPOUT = True
        if model is None:
            model = build_model(cfg)
        loader = build_detection_test_loader(cfg, pool_name)
        metadata = MetadataCatalog.get(pool_name)
        num_classes = len(metadata.thing_classes)
        metrics_dict = {}
        for data in loader:
            pred_boxes = [_forward_boxes(model, data) for t in range(self.T)]
            boxes = [box.pred_boxes for box in pred_boxes]
            scores = [box.scores.cpu().numpy() for box in pred_boxes]
            
            roi_matches = [[i]+[None for _ in range(self.T-1)] for i in range(len(boxes[0]))]
            for t in range(1, self.T):
                for i, box in enumerate(boxes[t]):
                    matched = None
                    max_iou = 0.0
                    for j, roim in enumerate(roi_matches):
                        for r, idx in enumerate(roim):
                            if idx is not None:
                                iou = self.get_iou(box, boxes[r][idx].tensor[0])
                                if iou > max_iou:
                                    max_iou = iou
                                    matched = j
                    if matched is None:
                        roi_matches.append([None for _ in range(t)] + [i] + [None for _ in range(self.T-t-1)])
                    else:
                        roi_matches[matched][t] = i
            uncertainties = []
            for matches in roi_matches:
                matched_scores = [score[match] for match, score in zip(matches, scores) if match is not None]
                uncertainty = self.get_uncertainty(matched_scores, num_classes)
                if uncertainty is not None:
                    uncertainties.append(uncertainty)
            
            metrics_dict[data[0]["image_id"]] = float(self.aggregation(uncertainties))
            
        rank = sorted(metrics_dict.items(), key=lambda x: x[1], reverse=True)    # Active Learning
        # rank = sorted(metrics_dict.items(), key=lambda x: x[1], reverse=False)    # Pseudo Labeling
        if verbose:
            headers = ['Selected files', 'Scores']
            contents = rank[:k]
            ctable(headers, contents, title="RoIMatchingStrategy.Selected", color='yellow')
        return [r[0] for r in rank[:k]]


def ctable(headers, contents, title=None, color='green'):
    '''
    Print colored table on terminal.
    Args:
        headers (list(str)): a list of string as the title of each column, e.g., ['Num1', 'Num2', 'Sum'].
        contents (list): a list of content in each row,e.g.,
            [[1, 1, 2],
             [2, 4, 6]].
    '''
    t = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    caller = sys._getframe(1).f_globals["__name__"] if title is None else str(title)
    title = colored("[{} {}]:".format(t, caller), color)
    table = tabulate(contents, headers, tablefmt="pretty")
    print(title, flush=True)
    print(table, flush=True)


def instance2mask(instances, colors):
    h, w = instances.image_size
    mask = np.zeros((h, w, 3), np.uint8)
    for c, m in zip(instances.pred_classes, instances.pred_masks):
        if colors[c]:
            m = cv2.cvtColor(m.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            mask = np.where(m==[True, True, True], colors[c], mask)
    return mask


def draw_masks(bg, masks):
    mask = bg.copy()    
    for m in masks:
        mask = np.where(m==[0, 0, 0], mask, m)
        
    return mask.astype(np.uint8)


class PseudoLabeling:
    def __init__(self, cfg=None, threshold=0.8, device='cuda', model_path="./pretrained_model/model_final.pth"):
        if cfg is None:
            cfg = get_cfg()
            cfg.thing_classes = ['intersection', 'spacing']
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
            cfg.DATALOADER.NUM_WORKERS = 2
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
            cfg.MODEL.WEIGHTS = model_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
            cfg.MODEL.DEVICE = device
        self.labels = ['Intersection', 'Spacing']
        self.colors = [(127, 0, 0), (0, 127, 0)]
        self.predictor = DefaultPredictor(cfg)
        self.visualizer = Visualizer
        
    def predict_mask(self, weighted):
        outputs = self.predictor(weighted)
        instances = outputs["instances"].to("cpu")
        mask = instance2mask(instances, self.colors)
        return mask, instances


def pseudo_labeling(cfg=None, anno=None, dataset_dir='./data', output=None, uncertainty_sample=None):
    '''
    Do pseudo labeling, as the title suggest.
    Args:
        cfg: cfg.
        anno (str): COCO annotation file!
        dataset_dir (str): Path to dataset includes color and depth folders
        output (dir): Annotation file output name
    '''

    assert anno is not None

    PseudoLabelingModule = PseudoLabeling(cfg=cfg)

    if output == None:
        output = anno

    with open(anno) as file:
        data = json.load(file)

    cocojson = []

    os.makedirs(os.path.join(dataset_dir, 'Masks-PL'), exist_ok=True)

    for content in data["images"]:
        if uncertainty_sample is None or content["id"] in uncertainty_sample:
            color = cv2.imread(os.path.join(dataset_dir, content["file_name"]))

            mask, instances = PseudoLabelingModule.predict_mask(color)

            _, fname = os.path.split(content["file_name"])
            cv2.imwrite(os.path.join(dataset_dir, 'Masks-PL/mask-{}'.format(fname)), mask)

            image_id = content["id"]

            cocojson.extend(instances_to_coco_json(instances, image_id))

    for n, content in enumerate(cocojson):
        content['id'] = n
        content['iscrowd'] = 0
        content['bbox'][0] = math.ceil(content['bbox'][0])
        content['bbox'][1] = math.ceil(content['bbox'][1])
        content['bbox'][2] = math.ceil(content['bbox'][2])
        content['bbox'][3] = math.ceil(content['bbox'][3])
        # content['area'] = content['bbox'][2] * content['bbox'][3]
        content['area'] = float(pycocotools.mask.area(content['segmentation']))
        # del content['score']
        
        mask = pycocotools.mask.decode(content['segmentation'])

        # Code source below: https://github.com/facebookresearch/Detectron/issues/100
        # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []

        for contour in contours:
            contour = contour.flatten().tolist()
            # segmentation.append(contour)
            if len(contour) > 4:
                segmentation.append(contour)
        if len(segmentation) == 0:
            continue
        
        content['segmentation'] = segmentation

    data['annotations'] = cocojson

    with open(output, "w") as f:
        json.dump(data, f)


def get_image_id(anno, uncertainty_sample):
    with open(anno) as file:
        data = json.load(file)

    for content in data["images"]:
        if content["id"] in uncertainty_sample:
            print(content["id"], content["file_name"])



def main():
    # 2019 U-girder
    register_coco_instances("2019_train", {}, "./2019_train/annotations_train.json", "./2019_train")
    register_coco_instances("2019_val", {}, "./2019_val/annotations_val.json", "./2019_val")
    # register_coco_instances("2019_da", {}, "./2019_da/annotations_train.json", "./2019_da")
    register_coco_instances("2019_da", {}, "./2019_da/annotations.json", "./2019_da")    # Before PL
    # register_coco_instances("2019_da_val", {}, "./2019_da_val/annotations_val.json", "./2019_da_val")

    # 2021 U-girder
    register_coco_instances("2021_test", {}, "./2021_test/annotations_test.json", "./2021_test")
    # register_coco_instances("2021_da", {}, "./2021_da/annotations_train.json", "./2021_da")
    register_coco_instances("2021_da", {}, "./2021_da/annotations.json", "./2021_da")
    # register_coco_instances("2021_da_val", {}, "./2021_da_val/annotations_val.json", "./2021_da_val")

    # 2021 Continuous Wall
    register_coco_instances("2021_wall_test", {}, "./2021_wall_test/annotations_test.json", "./2021_wall_test")
    # register_coco_instances("2021_wall_da", {}, "./2021_wall_da/annotations_train.json", "./2021_wall_da")
    register_coco_instances("2021_wall_da", {}, "./2021_wall_da/annotations.json", "./2021_wall_da")
    # register_coco_instances("2021_wall_da_val", {}, "./2021_wall_da_val/annotations_val.json", "./2021_wall_da_val")

    # register_coco_instances("batch_train", {}, "./batch_train/annotations_train.json", "./batch_train")
    # # register_coco_instances("batch_val", {}, "./2019_val/annotations_val.json", "./2019_val")
    # register_coco_instances("batch_val", {}, "./batch_val/annotations_val.json", "./batch_val")
    # register_coco_instances("batch_test", {}, "./batch_val/annotations_test.json", "./batch_test")

    register_coco_instances("2021_mrt_da", {}, "./2021_mrt_da/annotations.json", "./2021_mrt_da")
    register_coco_instances("2021_basin_da", {}, "./2021_basin_da/annotations.json", "./2021_basin_da")
    register_coco_instances("2021_rail_da", {}, "./2021_rail_da/annotations.json", "./2021_rail_da")
    register_coco_instances("2021_substation_da", {}, "./2021_substation_da/annotations.json", "./2021_substation_da")
    register_coco_instances("2021_sanying_da", {}, "./2021_sanying_da/annotations.json", "./2021_sanying_da")

    # Query Strategy Modules
    # TODO: Fix freeze & score always = -1.0000000826903714e-10 when using RoIMatching
    # RandomModule = RandomStrategy()
    UncertaintyModule = UncertaintyStrategy(measurement='entropy')
    # RoIModule = RoIMatchingStrategy()

    cfg = get_cfg()
    cfg.thing_classes = ['intersection', 'spacing']
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("2019_train",)
    cfg.DATASETS.TEST = ("2019_val",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.MODEL.WEIGHTS = "./pretrained_model/model_final.pth"
    cfg.MODEL.WEIGHTS = "./output-baseline/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = 'cuda'

    model = build_model(cfg)

    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    print("############ 2019 U-girder ############")
    evaluator = COCOEvaluator("2019_val", cfg, False, output_dir="./output-baseline/")
    val_loader = build_detection_test_loader(cfg, "2019_val")
    inference_on_dataset(model, val_loader, evaluator)

    print("############ 2021 U-girder ############")
    evaluator = COCOEvaluator("2021_test", cfg, False, output_dir="./output-baseline/")
    val_loader = build_detection_test_loader(cfg, "2021_test")
    inference_on_dataset(model, val_loader, evaluator)

    print("############ 2021 Diaphragm Wall ############")
    evaluator = COCOEvaluator("2021_wall_test", cfg, False, output_dir="./output-baseline/")
    val_loader = build_detection_test_loader(cfg, "2021_wall_test")
    inference_on_dataset(model, val_loader, evaluator)

    # uncertainty_sample_2019ugirder = UncertaintyModule.sample(cfg, "2019_da", k=280, model=model)    # 2019_da has 2796 photos
    # uncertainty_sample_2019ugirder = UncertaintyModule.sample(cfg, "2019_da", k=100, model=model)

    # anno = './2019_da/annotations.json'
    # get_image_id(anno, uncertainty_sample_2019ugirder)

    # random_sample_2021ugirder = RandomModule.sample(cfg, "2021_da", k=20, model=model)
    # uncertainty_sample_2021ugirder = UncertaintyModule.sample(cfg, "2021_da", k=172, model=model)    # 2021_da has 1570(1723?) photos
    # uncertainty_sample_2021ugirder = UncertaintyModule.sample(cfg, "2021_da", k=100, model=model)
    # roi_sample_2021ugirder = RoIModule.sample(cfg, "2021_da", k=20, model=model)    # TODO: Fix freeze when using RoIMatching

    # anno = './2021_da/annotations.json'
    # get_image_id(anno, uncertainty_sample_2021ugirder)

    # random_sample_2021wall = RandomModule.sample(cfg, "2021_wall_da", k=20, model=model)
    # uncertainty_sample_2021wall = UncertaintyModule.sample(cfg, "2021_wall_da", k=143, model=model)    # 2021_wall_da has 1427 photos
    # uncertainty_sample_2021wall = UncertaintyModule.sample(cfg, "2021_wall_da", k=100, model=model)
    # roi_sample_2021wall = RoIModule.sample(cfg, "2021_wall_da", k=20, model=model)

    # anno = './2021_wall_da/annotations.json'
    # get_image_id(anno, uncertainty_sample_2021wall)

    uncertainty_sample_2021mrt = UncertaintyModule.sample(cfg, "2021_mrt_da", k=50, model=model, skip=100)

    anno = './2021_mrt_da/annotations.json'
    get_image_id(anno, uncertainty_sample_2021mrt)

    uncertainty_sample_2021basin = UncertaintyModule.sample(cfg, "2021_basin_da", k=50, model=model, skip=60)

    anno = './2021_basin_da/annotations.json'
    get_image_id(anno, uncertainty_sample_2021basin)

    uncertainty_sample_2021rail = UncertaintyModule.sample(cfg, "2021_rail_da", k=50, model=model, skip=100)
    
    anno = './2021_rail_da/annotations.json'
    get_image_id(anno, uncertainty_sample_2021rail)

    uncertainty_sample_2021substation = UncertaintyModule.sample(cfg, "2021_substation_da", k=50, model=model, skip=60)

    anno = './2021_substation_da/annotations.json'
    get_image_id(anno, uncertainty_sample_2021substation)

    uncertainty_sample_2021sanying = UncertaintyModule.sample(cfg, "2021_sanying_da", k=50, model=model, skip=100)

    anno = './2021_sanying_da/annotations.json'
    get_image_id(anno, uncertainty_sample_2021sanying)


    # # TODO: Implement it with pseudo_labeling.py
    # anno = './2019_da/annotations.json'
    # dataset_dir = './2019_da'
    # output = './2019_da/annotations_train.json'
    # pseudo_labeling(cfg, anno, dataset_dir, output, uncertainty_sample_2019ugirder)

    # anno = './2021_da/annotations.json'
    # dataset_dir = './2021_da'
    # output = './2021_da/annotations_train.json'
    # pseudo_labeling(cfg, anno, dataset_dir, output, uncertainty_sample_2021ugirder)

    # anno = './2021_wall_da/annotations.json'
    # dataset_dir = './2021_wall_da'
    # output = './2021_wall_da/annotations_train.json'
    # pseudo_labeling(cfg, anno, dataset_dir, output, uncertainty_sample_2021wall)


if __name__ == '__main__':
    main()
