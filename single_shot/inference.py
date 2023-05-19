import argparse
# from ast import arg
import detectron2
from detectron2.data.datasets import register_coco_instances
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

import torch, torchvision

import numpy as np

from detectron2.utils.logger import setup_logger
import logging

from detectron2.engine import launch

from detectron2.utils.visualizer import ColorMode

from imantics import Mask
from scipy.spatial.distance import cdist
# from itertools import product

# from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

from detectron2.utils.visualizer import GenericMask

import pupil_apriltags as apriltag
from tqdm import tqdm
import time
import math

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

from shapely.geometry import LineString, Point, MultiLineString
from shapely import affinity

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

def main(args):
    start = time.time()

    # register_coco_instances("2019_train", {}, "../DA-MRCNN/2019_train/annotations_train.json", "../DA-MRCNN/2019_train")
    # register_coco_instances("2019_val", {}, "../DA-MRCNN/2019_val/annotations_val.json", "../DA-MRCNN/2019_val")
    # # register_coco_instances("2019_da", {}, "./2019_da/annotations_train.json", "./2019_da")
    # # register_coco_instances("2019_da_val", {}, "./2019_da_val/annotations_val.json", "./2019_da_val")

    # # register_coco_instances("2021_test", {}, "./2021_test/annotations_test.json", "./2021_test")
    # # register_coco_instances("2021_da", {}, "./2021_da/annotations_train.json", "./2021_da")
    # # register_coco_instances("2021_da_val", {}, "./2021_da_val/annotations_val.json", "./2021_da_val")

    setup_logger()
    logger = logging.getLogger("detectron2")

    cfg_source = get_cfg()
    cfg_source.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    # cfg_source.merge_from_file(model_zoo.get_config_file("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py"))
    # cfg_source.DATASETS.TRAIN = ("2019_train",)
    # cfg_source.DATASETS.TEST = ("2019_val",)
    cfg_source.DATALOADER.NUM_WORKERS = 2
    cfg_source.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    # cfg_source.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py")
    cfg_source.SOLVER.IMS_PER_BATCH = 4
    cfg_source.SOLVER.BASE_LR = 0.00005
    cfg_source.SOLVER.WEIGHT_DECAY = 0.001
    cfg_source.SOLVER.MAX_ITER = 30000
    cfg_source.SOLVER.STEPS = (300,)
    cfg_source.INPUT.MIN_SIZE_TRAIN = (0,)
    cfg_source.INPUT.MIN_SIZE_TEST = 0
    os.makedirs(cfg_source.OUTPUT_DIR, exist_ok=True)
    cfg_source.MODEL.ROI_HEADS.NUM_CLASSES = 2

    # cfg_source.MODEL.WEIGHTS = "../DA-MRCNN/pretrained_model/model_final.pth"
    cfg_source.MODEL.WEIGHTS = args.model_dir

    cfg_source.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    """ INFERENCE """
    predictor = DefaultPredictor(cfg_source)

    # test_path = "./RCB2_S1_0416_01/Raw"
    # test_path = "./stitched_image1"
    test_path = args.input_dir
    files = os.listdir(test_path)
    steel_metadata = MetadataCatalog.get("2019_train")
    # os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.verbose:
        os.makedirs(os.path.join(args.output_dir, "detailed_results"), exist_ok=True)

    end = time.time()
    print()
    print('################## Time for preparing model from detectron2:', end-start, 'seconds ##################')
    print()

    for f in files:
        if os.path.splitext(f)[-1].lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        start = time.time()

        print('###########################################################')
        print('################## Now processing:', f, '##################')
        print('###########################################################')

        image = cv2.imread(os.path.join(test_path, f))

        try_old_method = False
        try:
            length_per_pixel = np.load(os.path.join(test_path, f+'-length_per_pixel.npy'))
        except:
            try_old_method = True
        if try_old_method:
            try:
                coord_x = np.load(os.path.join(test_path, f+'-numpy_x_array.npy'))
                coord_y = np.load(os.path.join(test_path, f+'-numpy_y_array.npy'))
                length_per_pixel = coord_x[1] - coord_x[0]
            except:
                length_per_pixel = find_apriltag(image)

        image = cv2.resize(image, (1280, int(image.shape[0]/image.shape[1]*1280)))    # Resize image into 1280*720 to prevent CUDA OOM
        length_per_pixel = length_per_pixel / image.shape[1] * 1280

        im_list = [image]
        x_window_size = image.shape[1]
        y_window_size = image.shape[0]
        y_count = image.shape[0] // y_window_size + (image.shape[0] % y_window_size > 0)
        x_count = image.shape[1] // x_window_size + (image.shape[1] % x_window_size > 0)
        print("y_count:", y_count)
        print("x_count:", x_count)
        print("image.shape:", image.shape)
        print("np.shape(im_list):", np.shape(im_list))
        pred_classes_list = []
        pred_masks_list = []
        pred_boxes_list = []

        end = time.time()
        print()
        print('################## Time for preparing 1 image for inference:', end-start, 'seconds ##################')
        print()

        start = time.time()

        len_masks = 0

        for i, im in enumerate(tqdm(im_list)):
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                        metadata=steel_metadata, 
                        # scale=0.5, 
                        # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # cv2.imshow(out.get_image()[:, :, ::-1])
            if args.verbose:
                cv2.imwrite(os.path.join(args.output_dir, "detailed_results", f+"_mask_{}.jpg".format(i)), out.get_image()[:, :, ::-1])
            # mask_list.append(out.get_image()[:, :, ::-1])

            pred_masks = outputs['instances'].pred_masks
            pred_classes = outputs['instances'].pred_classes
            pred_boxes = outputs['instances'].pred_boxes

            # NOTE: I usually skip all the link prediction parts for cropped images, because I don't think they are useful & uses up a lot of inference time.
            if args.verbose:
                # junctions = find_centroid_by_mask(pred_masks[pred_classes==0])
                # enpoints, line_masks = find_endpoint_by_mask(pred_masks[pred_classes==1].cpu())
                junctions = find_centroid_by_box(pred_boxes[pred_classes==0].to(torch.device("cpu")))
                enpoints, line_masks = find_endpoint_by_box(pred_boxes[pred_classes==1].to(torch.device("cpu")))

                if enpoints.shape[0] > 0:
                    links = find_nearest_link(
                        junctions, enpoints,
                        # line_masks=line_masks,    # <BUG> THIS ONE IS BUGGYYYYYYYYYYYYYYYYYYYYYY
                        # max_e2j_dist=params.MODULE.SPACING_HEAD.E2J_THRESH,
                        # max_e2e_dist=params.MODULE.SPACING_HEAD.E2E_THRESH,
                        return_index=False).numpy().astype(int)

                    # coord_x = np.load('numpy_x_array.npy')
                    # coord_y = np.load('numpy_y_array.npy')

                    _reports = inspect_link(links, length_per_pixel)
                    PASS, _groups = ActiveInspection(args, _reports, f, length_per_pixel)
                    _groups["horizontal"] = find_link_group(im.shape[1], _groups["horizontal"])
                    _groups["vertical"] = find_link_group(im.shape[0], _groups["vertical"])
                    _visualization = vis_group(im, _groups)
                    cv2.imwrite(os.path.join(args.output_dir, "detailed_results", f+"_link_{}.jpg".format(i)), _visualization)
                    # link_list.append(_visualization)

            pred_boxes.tensor[:, 0::2] += x_window_size * (i % x_count)
            pred_boxes.tensor[:, 1::2] += y_window_size * (i // x_count)
            # print("pred_masks:", pred_masks)
            masks = np.asarray(pred_masks.cpu())
            # print("pred_masks.shape:", masks.shape)
            masks = [GenericMask(x, im.shape[0], im.shape[1]).polygons for x in masks]
            # print("masks:", masks)
            # print("len(masks):", len(masks))
            len_masks += len(masks)
            # print("masks[0][0].shape:", masks[0][0].shape)
            for mask in masks:
                for polygon in mask:
                    if polygon.shape[0] < 8:
                        print("WARNING: polygon.shape =", polygon.shape)
                    # if polygon.shape[0] < 8:
                    #     del polygon
                    polygon[0::2] += x_window_size * (i % x_count)
                    polygon[1::2] += y_window_size * (i // x_count)

            if len(masks) > 0:
                pred_classes_list.append(pred_classes)
                pred_boxes_list.append(pred_boxes)
                pred_masks_list += masks

        end = time.time()
        print()
        print('################## Time for inferencing the cropped images:', end-start, 'seconds ##################')
        print()

        start = time.time()

        # mask_list = np.asarray(mask_list).reshape((y_count, x_count))

        print("len(pred_masks_list):", len(pred_masks_list))
        if len_masks > 1:
            pred_classes_final = torch.cat(pred_classes_list, dim=0)
            pred_boxes_final = Boxes.cat(pred_boxes_list)
            pred_boxes_final.clip((image.shape[1], image.shape[0]))
            assert len(pred_masks_list)  == len_masks
            assert len(pred_classes_final) == len(pred_masks_list)

            pred_masks_final = np.asarray([GenericMask(x, image.shape[0], image.shape[1]) for x in pred_masks_list])


            print("len(pred_masks_final) AFTER CLEANING:", len(pred_masks_final))
            # assert len(pred_masks_list)  == len_masks
            assert len(pred_classes_final) == len(pred_masks_final)

            print('image shape:', image.shape)
            print('image shape[1]:', image.shape[1]) # x
            print('image shape[0]:', image.shape[0]) # y


            # junctions = find_centroid_by_polygon(pred_masks_final[pred_classes_final.cpu()==0])
            # enpoints, line_masks = find_endpoint_by_polygon(pred_masks_final[pred_classes_final.cpu()==1])
            junctions = find_centroid_by_box(pred_boxes_final[pred_classes_final==0].to(torch.device("cpu")))
            enpoints, line_masks = find_endpoint_by_box(pred_boxes_final[pred_classes_final==1].to(torch.device("cpu")))

            if enpoints.shape[0] > 0:
                links = find_nearest_link(
                    junctions, enpoints,
                    line_masks=line_masks,
                    # max_e2j_dist=params.MODULE.SPACING_HEAD.E2J_THRESH,
                    # max_e2e_dist=params.MODULE.SPACING_HEAD.E2E_THRESH,
                    max_e2j_dist=50,
                    # max_e2e_dist=300,
                    return_index=False).numpy().astype(int)

                _reports = inspect_link(links, length_per_pixel)

                PASS, _groups = ActiveInspection(args, _reports, f, length_per_pixel)
                _groups["horizontal"] = find_link_group(image.shape[1], _groups["horizontal"])
                _groups["vertical"] = find_link_group(image.shape[0], _groups["vertical"])
                # print("_groups[horizontal]:", _groups["horizontal"])
                # print("_groups[vertical]:", _groups["vertical"])
                # _visualization = vis_link(image, _reports)
                _visualization = vis_group(image, _groups)
                cv2.imwrite(os.path.join(args.output_dir, f+"_link_stitched.jpg"), _visualization)
                _visualization_rotated = vis_group_rotated(image, _groups)
                cv2.imwrite(os.path.join(args.output_dir, f+"_link_rotated.jpg"), _visualization_rotated)

                _group_horizontal = sort_by_groupid(_groups["horizontal"], "horizontal")
                _group_vertical = sort_by_groupid(_groups["vertical"], "vertical")

                _group_horizontal = vis_link_group(image, _group_horizontal, args, f, "horizontal")
                # image is produced within vis_link_group() function
                if length_per_pixel is not None:
                    result_horizontal = process_group(_group_horizontal, _group_vertical, length_per_pixel)
                    result_horizontal = print_inspection_report(args, f, result_horizontal, "horizontal")
                    # print("Horizontal line inspection result:", result_horizontal)
                    vis_group_spacing(image, _group_horizontal, result_horizontal, args, f, "horizontal")
                
                _group_vertical = vis_link_group(image, _group_vertical, args, f, "vertical")
                # Notice the order: vis_link_group() -> vis_group_spacing() -> NEXT GRAPH
                if length_per_pixel is not None:
                    result_vertical = process_group(_group_vertical, _group_horizontal, length_per_pixel)
                    result_vertical = print_inspection_report(args, f, result_vertical, "vertical")
                    # print("Vertical line inspection result:", result_vertical)
                    vis_group_spacing(image, _group_vertical, result_vertical, args, f, "vertical")

        end = time.time()
        print()
        print('################## Time for combining the cropped images:', end-start, 'seconds ##################')
        print()



def print_inspection_report(args, filename, result, direction, decision=None):
    start_time = time.time()
    print("Producing inspection report...")
    num_rebars = len(result) + 1
    mean_spacings = 0
    mean_spacing_list = []
    median_spacings = 0
    median_spacing_list = []
    result_key = list(result.keys())
    error_count = 0
    if decision == 'mean':
        for spacing in result.values():
            # mean_spacings += spacing['mean']
            mean_spacing_list.append(spacing['mean'])
        # mean_spacings /= (num_rebars - 1)
        mean_spacings = np.mean(mean_spacing_list)
        for i, spacing in enumerate(mean_spacing_list):
            if abs(spacing - mean_spacings) > args.tolerance_bias or abs(spacing - mean_spacings)/mean_spacings > args.tolerance_ratio:
                error_count += 1
                result[result_key[i]]["outliers"] = True
            else:
                result[result_key[i]]["outliers"] = False
    elif decision == 'median':
        for spacing in result.values():
            # median_spacings += spacing['median']
            median_spacing_list.append(spacing['median'])
        # median_spacings /= (num_rebars - 1)
        median_spacings = np.median(median_spacing_list)
        for i, spacing in enumerate(median_spacing_list):
            if abs(spacing - median_spacings) > args.tolerance_bias or abs(spacing - median_spacings)/median_spacings > args.tolerance_ratio:
                error_count += 1
                result[result_key[i]]["outliers"] = True
            else:
                result[result_key[i]]["outliers"] = False
    else:
        print('Cannot decide which method to use for averaging spacing? I will provide both.')
        for spacing in result.values():
            # mean_spacings += spacing['mean']
            mean_spacing_list.append(spacing['mean'])
            # median_spacings +=  spacing['median']
            median_spacing_list.append(spacing['median'])
        # mean_spacings /= (num_rebars - 1)
        # median_spacings /= (num_rebars - 1)
        mean_spacings = np.mean(mean_spacing_list)
        median_spacings = np.median(median_spacing_list)
        for i, (mean, median) in enumerate(zip(mean_spacing_list, median_spacing_list)):
            condition_mean = abs(mean - mean_spacings) > args.tolerance_bias or abs(mean - mean_spacings)/mean_spacings > args.tolerance_ratio
            condition_median = abs(median - median_spacings) > args.tolerance_bias or abs(median - median_spacings)/median_spacings > args.tolerance_ratio
            if condition_mean or condition_median:
                error_count += 1
                result[result_key[i]]["outliers"] = True
            else:
                result[result_key[i]]["outliers"] = False
    
    report_file = os.path.join(args.output_dir, "inspection_report.txt")
    with open(report_file, 'a') as f:
        f.write('\n##########################################################################################')
        f.write('\n################## INSPECTION REPORT: {} (Direction: {}) ##################'.format(filename, direction))
        f.write('\n##########################################################################################')
        f.write('\n')
        f.write('\nNumber of rebars: {}'.format(num_rebars))
        if mean_spacings:
            f.write('\nMean length of rebar spacings: {:.2f} cm'.format(mean_spacings))
        if median_spacings:
            f.write('\nMedian length of rebar spacings: {:.2f} cm'.format(median_spacings))
        f.write('\nNumber of non-standard* spacings: {:d}'.format(error_count))
        f.write('\n(*Standard: error within 5 cm or 20 %)')
        f.write('\n')

    end_time = time.time()
    print("Time for producing inspection report:", end_time-start_time, "seconds")
    return result

def find_apriltag(one_image, apriltag_size = 0.18):    # apriltag_size in meters
    w_downsample = 2400
    resize_tuple = (w_downsample, int(one_image.shape[0] * w_downsample / one_image.shape[1]))
    # print(resize_tuple)
    img_resize = cv2.resize(one_image, resize_tuple, interpolation=cv2.INTER_LINEAR_EXACT)
    # img_resize = one_image
    gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    tag_detector = apriltag.Detector(families='tag36h11', # tag25h9
                                    #  border=1,
                                        nthreads=2,
                                        quad_decimate=1.0,
                                    #  quad_blur=3,
                                    #  refine_edges=True,
                                    #  refine_decode=True,
                                    #  refine_pose=False,
                                    #  debug=False,
                                    #  quad_contours=True
                                    )
        
    # Perform apriltag detection to get a list of detected apriltag
    # tags = tag_detector.detect(gray_image, estimate_tag_pose=False) 
    cam_params = [2.46634964e+03, 2.59200000e+03, 1.94400000e+03, 5.46740775e-04]
    # cam_params = [1000, 2000, 2000, 5.46740775e-04]
    tags = tag_detector.detect(gray_image, estimate_tag_pose=True, camera_params = cam_params, tag_size=apriltag_size) 
    if len(tags) == 0: 
        print("Skipping since no AprilTag detected.")
        return None

    for i, tag in enumerate(tags):
        print()
        print('############ TAG', i, '############')
        print(tag)

    one_tag = tags[0]
    c0 = (one_tag.corners[0]).astype(int)[: : -1] / w_downsample * one_image.shape[1]
    c1 = (one_tag.corners[1]).astype(int)[: : -1] / w_downsample * one_image.shape[1] 
    c2 = (one_tag.corners[2]).astype(int)[: : -1] / w_downsample * one_image.shape[1]
    c3 = (one_tag.corners[3]).astype(int)[: : -1] / w_downsample * one_image.shape[1]

    pixel_distance = 0.25*(np.sum((c0-c1)**2)**0.5+np.sum((c1-c2)**2)**0.5+np.sum((c2-c3)**2)**0.5+np.sum((c3-c0)**2)**0.5)
    lengh_per_pixel = apriltag_size/pixel_distance
    print('Length per pixel:', lengh_per_pixel)
    return lengh_per_pixel

def find_centroid_by_box(boxes):
    """
    Find centroids from predicted boxes.
    Args:
        boxes (torch.tensor): a Nx4 matrix. Each row is (x1, y1, x2, y2).
    Returns:
        The box centers in a Nx2 array of (x, y).
    """
    return boxes.get_centers()

def find_endpoint_by_box(boxes, return_linemasks=True):
    """
    Find the two farest points as the endpoints of the each box.
    Args:
        boxes (torch.tensor): a Nx4 matrix. Each row is (x1, y1, x2, y2).
        return_linemasks (bool): if return the masks with endpoints.
    Returns:
        torch tensors
    """
    endpoints = []
    linemask_indices = []
    for m, box in enumerate(boxes):
        pts = np.asarray(box.to(torch.device("cpu")))
        endpoints.append([[pts[0], pts[1]], [pts[2], pts[3]]])
        linemask_indices.append(m)
    if return_linemasks:
        # return torch.tensor(endpoints), masks[linemask_indices]
        return torch.tensor(endpoints), None    # TODO: make lineboxes in format of linemasks (but gonna need to adjust find_nearest_links())
    return torch.tensor(endpoints)

def find_centroid_by_polygon(masks):
    """
    Find centroids from predicted mask polygon.
    Args:
        masks (torch.tensor): a tensor of predicted masks of shape (N, H, W), where N is the number
            of masks. H and W is the height and width of the mask, respectively.
    Returns:
        a torch tensor
    """
    centroids = []
    for mask in np.asarray(masks):
        polys = mask.polygons
        pts = [pt for contour in polys for pt in contour.reshape(-1, 2)]
        if len(pts) >= 3:
            M = cv2.moments(np.array(pts))
            centroids.append([M['m10'] / M["m00"], M['m01'] / M["m00"]])
    return torch.tensor(centroids)

def find_endpoint_by_polygon(masks, return_linemasks=True):
    """
    Find the two farest points as the endpoints of the each mask polygon.
    Args:
        masks (torch.tensor): a tensor of predicted masks of shape (N, H, W), where N is the number
            of mask. H and W is the height and width of the mask, respectively
        return_linemasks (bool): if return the masks with endpoints.
    Returns:
        torch tensors
    """
    endpoints = []
    linemask_indices = []
    for m, mask in enumerate(np.asarray(masks)):
        polys = mask.polygons
        pts = [pt for contour in polys for pt in contour.reshape(-1, 2)]
        if len(pts) >= 2:
            dist_matrix = cdist(pts, pts, metric='euclidean')
            i, j = np.where(dist_matrix==dist_matrix.max())[0][:2]
            endpoints.append([pts[i], pts[j]])
            linemask_indices.append(m)
    if return_linemasks:
        # return torch.tensor(endpoints), masks[linemask_indices]
        return torch.tensor(endpoints), None    # NOTE: Line mask is bugged also
    return torch.tensor(endpoints)

def find_centroid_by_mask_new(masks):
    """
    Find centroids from predicted masks.
    Args:
        masks (torch.tensor): a tensor of predicted masks of shape (N, H, W), where N is the number
            of masks. H and W is the height and width of the mask, respectively.
    Returns:
        a torch tensor
    """
    centroids = []
    for mask in np.asarray(masks):
        mask = mask.mask
        polys = Mask(mask).polygons()
        pts = [pt for contour in polys.points for pt in contour]
        if len(pts) >= 3:
            M = cv2.moments(np.array(pts))
            centroids.append([M['m10'] / M["m00"], M['m01'] / M["m00"]])
    return torch.tensor(centroids)

def find_endpoint_by_mask_new(masks, return_linemasks=True):
    """
    Find the two farest points as the endpoints of the each mask.
    Args:
        masks (torch.tensor): a tensor of predicted masks of shape (N, H, W), where N is the number
            of mask. H and W is the height and width of the mask, respectively
        return_linemasks (bool): if return the masks with endpoints.
    Returns:
        torch tensors
    """
    endpoints = []
    linemask_indices = []
    for m, mask in enumerate(np.asarray(masks)):
        mask = mask.mask
        polys = Mask(mask).polygons()
        pts = [pt for contour in polys.points for pt in contour]
        if len(pts) >= 2:
            dist_matrix = cdist(pts, pts, metric='euclidean')
            i, j = np.where(dist_matrix==dist_matrix.max())[0][:2]
            endpoints.append([pts[i], pts[j]])
            linemask_indices.append(m)
    if return_linemasks:
        # return torch.tensor(endpoints), masks[linemask_indices]
        return torch.tensor(endpoints), None    # NOTE: Line mask is bugged also
    return torch.tensor(endpoints)

def find_centroid_by_mask(masks):
    """
    Find centroids from predicted masks.
    Args:
        masks (torch.tensor): a tensor of predicted masks of shape (N, H, W), where N is the number
            of masks. H and W is the height and width of the mask, respectively.
    Returns:
        a torch tensor
    """
    centroids = []
    for mask in np.asarray(masks.cpu()):
        polys = Mask(mask).polygons()
        pts = [pt for contour in polys.points for pt in contour]
        if len(pts) >= 3:
            M = cv2.moments(np.array(pts))
            centroids.append([M['m10'] / M["m00"], M['m01'] / M["m00"]])
    return torch.tensor(centroids)

def find_endpoint_by_mask(masks, return_linemasks=True):
    """
    Find the two farest points as the endpoints of the each mask.
    Args:
        masks (torch.tensor): a tensor of predicted masks of shape (N, H, W), where N is the number
            of mask. H and W is the height and width of the mask, respectively
        return_linemasks (bool): if return the masks with endpoints.
    Returns:
        torch tensors
    """
    endpoints = []
    linemask_indices = []
    for m, mask in enumerate(np.asarray(masks.cpu())):
        polys = Mask(mask).polygons()
        pts = [pt for contour in polys.points for pt in contour]
        if len(pts) >= 2:
            dist_matrix = cdist(pts, pts, metric='euclidean')
            i, j = np.where(dist_matrix==dist_matrix.max())[0][:2]
            endpoints.append([pts[i], pts[j]])
            linemask_indices.append(m)
    if return_linemasks:
        return torch.tensor(endpoints), masks[linemask_indices]
    return torch.tensor(endpoints)

def find_nearest_link(juncs, lines, line_masks=None,
        max_e2j_dist=30, max_e2e_dist=50, path_thred=0.5, e2e_on=True, return_index=True):
    """
    Find the links between junctions and lines.
    Args:
        juncs (torch.tensor): a tensor of junctions of shape (N, 2), where N is the number
            of junction. Each junction is represented by a point (X, Y).
        lines (torch.tensor): a tensor of lines of shape (N, 2, 2), where N is the number
            of line. Each line is represented by two points.
        line_masks (Optional[torch.tensor]): a tensor of predicted masks of shape (N, H, W), where N is the number
            of mask. H and W is the height and width of the mask, respectively
        max_e2j_dist (int): the maximun tolerance distance between endpoints and junctions.
        max_e2e_dist (int): the maximun tolerance distance between endpoints and enpoints.
        path_thred (Optional[float]): a float between [0, 1] that filters out links with path confindence under path_thred.
        return_index (bool): if return the indices of connected junction.
    Returns:
        a torch tensor
    """
    def line_line_intersection(line1, line2):
        v1 = line1[1]-line1[0]
        v2 = line2[1]-line2[0]
        m1 = v1[1] / v1[0]
        m2 = v2[1] / v2[0]
        if m1 != m2:
            x = (line2[0, 1]-line1[0, 1] + m1 * line1[0, 0] - m2 * line2[0, 0]) / (m1 - m2)
            y = m1 * (x - line1[0, 0]) + line1[0, 1]
            intersection = torch.tensor([x, y])
            if torch.linalg.norm(intersection-line1[1]) < max_e2j_dist and torch.linalg.norm(intersection-line2[1]) < max_e2j_dist:
                return intersection
        
    links = []
    for l, line in enumerate(lines):
        # E2J link prediction
        if juncs.shape[0] > 0:
            e2j_dist_matrix = cdist(line, juncs, metric='euclidean')
            i, j = e2j_dist_matrix.argsort(1)[:,0]
            if i != j and e2j_dist_matrix[0, i] < max_e2j_dist and e2j_dist_matrix[1, j] < max_e2j_dist:
                if line_masks is not None:
                    length = torch.linalg.norm(juncs[i]-juncs[j]).int()
                    x = torch.linspace(juncs[i][0], juncs[j][0], length).long()
                    y = torch.linspace(juncs[i][1], juncs[j][1], length).long()
                    if line_masks[l, y, x].sum() / length < path_thred:
                        continue
                        
                if return_index:
                    links.append([i, j])
                    
                else:
                    links.append(juncs[[i, j]].numpy().tolist())
        
        # E2E link prediction
        elif e2e_on:
            dist_ei, dist_ej = cdist(line, lines.view(-1, 2), metric='euclidean')
            if e2j_dist_matrix[0, i] < max_e2j_dist:
                for ej in np.where((0 < dist_ej) & (dist_ej < max_e2e_dist))[0]:
                    intersection = line_line_intersection(line, lines[ej // 2, [1 - ej % 2, ej % 2]])
                    if intersection is not None:
                        juncs = torch.cat([juncs, intersection.unsqueeze(0)], 0)
                        if return_index:
                            links.append([i, len(juncs)-1])
                            
                        else:
                            links.append(juncs[[i, -1]].numpy().tolist())
                        break
                    
            elif e2j_dist_matrix[1, j] < max_e2j_dist:
                for ei in np.where((0 < dist_ei) & (dist_ei < max_e2e_dist))[0]:
                    intersection = line_line_intersection(line[[1, 0]], lines[ei // 2, [1 - ei % 2, ei % 2]])
                    if intersection is not None:
                        juncs = torch.cat([juncs, intersection.unsqueeze(0)], 0)
                        if return_index:
                            links.append([j, len(juncs)-1])
                            
                        else:
                            links.append(juncs[[j, -1]].numpy().tolist())
                        break
            else:
                link = []
                for ej in np.where((0 < dist_ej) & (dist_ej < max_e2e_dist))[0]:
                    intersection = line_line_intersection(line, lines[ej // 2, [1 - ej % 2, ej % 2]])
                    if intersection is not None:
                        juncs = torch.cat([juncs, intersection.unsqueeze(0)], 0)
                        link.append(len(juncs)-1)
                        break
                for ei in np.where((0 < dist_ei) & (dist_ei < max_e2e_dist))[0]:
                    intersection = line_line_intersection(line[[1, 0]], lines[ei // 2, [1 - ei % 2, ei % 2]])
                    if intersection is not None:
                        juncs = torch.cat([juncs, intersection.unsqueeze(0)], 0)
                        link.append(len(juncs)-1)
                        break
                if len(link) == 2:
                    if return_index:
                        links.append(link)
                    else:
                        links.append(juncs[link].numpy().tolist())
    if return_index:
        return juncs, torch.tensor(links)
        
    return torch.tensor(links)

def inspect_link_old(links, coord_x, coord_y):    # Deprecated.
    """
    Args:
        links (numpy.ndarray): an array of two endpoints, i.e., numpy.ndarray([[[0, 0], [10, 10]], [[0, 10], [50, 10]], ...])
        coord (numpy.ndarray): an 3d coordinates of shape (H, W, 3), where H and W is the height and width of the mask, respectively
    Returns:
        a dict of inspection reports
    """
    reports = {"statistics": {}, "spacing": {}}

    reports['statistics'] = {
        'num_spacing': len(links)
    }
    for i, link in enumerate(links):
        (x1, y1), (x2, y2) = link
        length = None
        if coord_x is not None:
            coord1 = np.array([coord_y[y1], coord_x[x1]])
            coord2 = np.array([coord_y[y2], coord_x[x2]])
            if np.sum(coord1 != 0) and np.sum(coord2 != 0):
                length = float(np.linalg.norm(coord1-coord2) * 100)
            dpx, dpy = x1-x2, y1-y2
            if dpx: 
                # theta = abs(math.atan(dpy / dpx) * 180 / math.pi)
                theta = math.atan(dpy / dpx) * 180 / math.pi
                if theta < -45:
                    theta += 180
            else: theta = 90
        if length > 0:
            reports['spacing']['S'+str(i)] = {
                "points": [[int(x1), int(y1)], [int(x2), int(y2)]],
                "length": length,
                "orientation": theta,
                "location": (int((x1+x2)/2), int((y1+y2)/2))
            }
    return reports

def inspect_link(links, length_per_pixel=None):
    """
    Args:
        links (numpy.ndarray): an array of two endpoints, i.e., numpy.ndarray([[[0, 0], [10, 10]], [[0, 10], [50, 10]], ...])
        length_per_pixel (float): actual length per pixel, calculated by apriltag
    Returns:
        a dict of inspection reports
    """
    reports = {"statistics": {}, "spacing": {}}

    reports['statistics'] = {
        'num_spacing': len(links)
    }
    for i, link in enumerate(links):
        (x1, y1), (x2, y2) = link
        length = None
        if length_per_pixel is not None:
            length = float(np.linalg.norm(link[0]-link[1]) * length_per_pixel * 100)
        dpx, dpy = x1-x2, y1-y2
        if dpx:
            # theta = abs(math.atan(dpy / dpx) * 180 / math.pi)
            theta = math.atan(dpy / dpx) * 180 / math.pi
            if theta < -45:
                theta += 180
        else: theta = 90
        # if length > 0:
        reports['spacing']['S'+str(i)] = {
            "points": [[int(x1), int(y1)], [int(x2), int(y2)]],
            "length": length,
            "orientation": theta,
            "location": (int((x1+x2)/2), int((y1+y2)/2))
        }
    return reports

def ActiveInspection(args, info, filename, length_per_pixel=None):
    # name = info['name']
    info = info['spacing']

    # Grouping spacings

    # # pairs = [IF['id'] for IF in info]
    # pairs = [tuple(map(tuple, IF['points'])) for IF in info.values()]
    # groups = {}
    # for (x, y) in pairs:
    #     xset = groups.get(x, set([x]))
    #     yset = groups.get(y, set([y]))
    #     jset = xset | yset
    #     for z in jset:
    #         groups[z] = jset
    # groups = set(map(tuple, groups.values()))
    # # for i in range(len(info)):        
    # for i in info:
    #     for j, group in enumerate(groups):            
    #         # if info[i]['id'][0] in group or info[i]['id'][1] in group:
    #         if tuple(info[i]['points'][0]) in group or tuple(info[i]['points'][1]) in group:
    #             info[i]['group_id'] = j
    
    # <NOTE> SJ uses "group" as separation of rebar structures, but I change it to as separation of vertical and horizontal rebars.
    # They might be able to work together. Can be a future work.

    orientations = [IF['orientation'] for IF in info.values()]

    horizontal_orientation, horizontal_std, vertical_orientation, vertical_std = double_gaussian(args, orientations, filename)

    vertical_group = []
    horizontal_group = []
    outlier_group = []
    for orien in orientations:
        if orien > horizontal_orientation - 2 * horizontal_std and orien < horizontal_orientation + 2 * horizontal_std:     # Check 68-95-99.7 rule. I use 2 sigma = 95%
            horizontal_group.append(orien)
        elif orien > vertical_orientation - 2 * vertical_std and orien < vertical_orientation + 2 * vertical_std:    # Check 68-95-99.7 rule. I use 2 sigma = 95%
            vertical_group.append(orien)
        else:
            outlier_group.append(orien)
    for i in info:
        if info[i]["orientation"] in horizontal_group:
            info[i]["direction"] = "horizontal"
            rotated_points = affinity.rotate(LineString(info[i]["points"]), horizontal_orientation - info[i]["orientation"], info[i]["location"]).coords[:]
            info[i]["rotated_points"] = np.asarray(rotated_points)
            info[i]["rotated_points_int"] = np.asarray(rotated_points).astype(int)
        elif info[i]["orientation"] in vertical_group:
            info[i]["direction"] = "vertical"
            rotated_points = affinity.rotate(LineString(info[i]["points"]), vertical_orientation - info[i]["orientation"], info[i]["location"]).coords[:]
            info[i]["rotated_points"] = np.asarray(rotated_points)
            info[i]["rotated_points_int"] = np.asarray(rotated_points).astype(int)
        elif info[i]["orientation"] in outlier_group:
            info[i]["direction"] = "outlier"
            info[i]["rotated_points"] = None
            info[i]["rotated_points_int"] = None
        else:
            print("Some mysterious error occured when finding the direction of the orientation of spacings!")
            print("info[i][orientation]:", info[i]["orientation"])

    print("len(horizontal_group):", len(horizontal_group))
    print("len(vertical_group):", len(vertical_group))
    print("len(outlier_group):", len(outlier_group))

    # Sorting orientation
    # groups = {i: [] for i in range(len(groups))}
    groups = {"horizontal": [], "vertical": [], "outlier": []}
    for IF in info.values():
        groups[IF['direction']].append({
            'points': IF['points'], 'length': IF['length'], 'orientation': IF['orientation'], 'location': IF['location'], 
            'rotated_points': IF['rotated_points'], 'rotated_points_int': IF['rotated_points_int'], 'pass': True, 'group_id': -1
            })
    for gi, group in groups.items():
        if gi == "horizontal":
            groups[gi] = sorted(group, key = lambda k: k['location'][1])
        elif gi == "vertical":
            groups[gi] = sorted(group, key = lambda k: k['location'][0])
    
    # Active inspection
    PASS = True

    # if groups["horizontal"][0]["length"] is not None:
    if length_per_pixel is not None:
        for gi, group in groups.items():
            split_idx = 0
            # for i in range(len(group)-1):
            #     if abs(group[i]['orientation'] - group[i+1]['orientation']) > 45:
            #         split_idx = i + 1
            #         break
            
            if split_idx:
                if len(group[:split_idx]) > args.min_num_for_active_inspection:
                    # pseudo_gt = np.mean([info['length'] for info in group[:split_idx]])
                    pseudo_gt = np.median([info['length'] for info in group[:split_idx]])
                    for j, info in enumerate(group[:split_idx]):
                        bias = abs(groups[gi][j]['length']-pseudo_gt)
                        if bias > args.tolerance_bias and bias > pseudo_gt * args.tolerance_ratio:
                            groups[gi][j]['pass'] = False
                            (x1, y1), (x2, y2) = groups[gi][j]['points']                            
                            PASS = False
                            
                if len(group[split_idx:]) > args.min_num_for_active_inspection:
                    # pseudo_gt = np.mean([info['length'] for info in group[split_idx:]])
                    pseudo_gt = np.median([info['length'] for info in group[split_idx:]])
                    for j, info in enumerate(group[split_idx:]):
                        bias = abs(groups[gi][split_idx+j]['length']-pseudo_gt)
                        if bias > args.tolerance_bias and bias > pseudo_gt * args.tolerance_ratio:
                            groups[gi][split_idx+j]['pass'] = False
                            (x1, y1), (x2, y2) = groups[gi][split_idx+j]['points']                            
                            PASS = False
                        
            else:
                if len(group) > args.min_num_for_active_inspection:
                    # pseudo_gt = np.mean([info['length'] for info in group])
                    pseudo_gt = np.median([info['length'] for info in group])
                    for i, info in enumerate(group):
                        bias = abs(groups[gi][i]['length']-pseudo_gt)
                        if bias > args.tolerance_bias and bias > pseudo_gt * args.tolerance_ratio:
                            groups[gi][i]['pass'] = False
                            (x1, y1), (x2, y2) = groups[gi][i]['points']                            
                            PASS = False

    # print("groups after active inspection:", groups)
    return PASS, groups

######## Source of fitting 2 gaussian distributions: https://stackoverflow.com/questions/35990467/ ########

def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodel(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def double_gaussian(args, data, filename):
    plt.close()
    plt.figure()
    # y, x, _ = plt.hist(np.asarray(data), max(100, len(data)), alpha=.3, label='data')
    y, x, _ = plt.hist(np.asarray(data), max(len(data)//2, 1), alpha=.3, label='data')
    plt.savefig(os.path.join(args.output_dir, filename + '-spacing_orientation_histogram.jpg'))
    x = (x[1:] + x[:-1]) / 2 # for len(x)==len(y)

    expected = (0, 5, 250, 90, 5, 250)
    try:
        params, cov = curve_fit(bimodel, x, y, expected)
        sigma = np.sqrt(np.diag(cov))

        x_fit = np.linspace(x.min(), x.max(), 500)
        #plot combined...
        plt.plot(x_fit, bimodel(x_fit, *params), color='red', lw=3, label='model')
        #...and individual Gauss curves
        plt.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=1, ls="--", label='distribution 1')
        plt.plot(x_fit, gauss(x_fit, *params[3:]), color='red', lw=1, ls=":", label='distribution 2')
        #and the original data points if no histogram has been created before
        #plt.scatter(x, y, marker="X", color="black", label="original data")
        plt.legend()
        print(pd.DataFrame(data={'params': params, 'sigma': sigma}, index=bimodel.__code__.co_varnames[1:]))
        # plt.show() 
        plt.savefig(os.path.join(args.output_dir, filename + '-double_gaussian_distribution.jpg'))
    except:
        params = expected
        print("Optimal parameters not found in double gaussian. Default to 0 and 90 degrees.")

    return params[0], params[1], params[3], params[4]

def find_link_group(dimension, group, max_e2e_dist=30, min_e2e_dist=30, search_limit=None):
    """input format: group["direction"], which should be a list of dicts"""
    if not search_limit:
        search_limit = max(100, len(group)//2)
    print("search_limit:", search_limit)
    i = 0
    j = 1
    group_id = 0
    group_id_dict = dict()
    new_group_id = False
    search_fail_time = 0
    # initial = True
    while i < len(group) - 1:
        a0, a1 = group[i]["rotated_points"]
        # b0, b1 = group[j]["rotated_points"]
        pt = group[j]["location"]
        # pt0, pt1, dist = closestDistanceBetweenLines(a0, a1, b0, b1)
        perp_dist, real_dist = distance_btw_line_and_pt(a0, a1, pt)
        if perp_dist < max(min_e2e_dist, max_e2e_dist - (max_e2e_dist - min_e2e_dist) * real_dist / dimension):
            if group[i]["group_id"] < 0:
                    if group[j]["group_id"] >= 0:
                        group[i]["group_id"] = group[j]["group_id"]
                        group_id_dict[group[j]["group_id"]].add(i)
                    else:
                        group[i]["group_id"] = group_id
                        group[j]["group_id"] = group_id
                        new_group_id = True
                        group_id_dict[group_id] = {i, j}
            else:
                if group[j]["group_id"] < 0:
                    group[j]["group_id"] = group[i]["group_id"]
                    group_id_dict[group[i]["group_id"]].add(j)
                elif group[i]["group_id"] != group[j]["group_id"]:
                    print("Special condition occurs! Group {} and group {} should be connected. perp_dist: {}, real_dist: {}".format(group[i]["group_id"], group[j]["group_id"], perp_dist, real_dist))
                    # print("group_id_dict[group[{}][group_id]] before operation:".format(i), group_id_dict[group[i]["group_id"]])
                    # print("group_id_dict[group[{}][group_id]] before operation:".format(j), group_id_dict[group[j]["group_id"]])
                    old_group_id = group[i]["group_id"]
                    temp_group_id = group[j]["group_id"]
                    group_id_dict[temp_group_id].update(group_id_dict[old_group_id])
                    for index in group_id_dict[old_group_id]:
                        # print(index)
                        group[index]["group_id"] = temp_group_id
                    group_id_dict.pop(old_group_id)
                    # print("group_id_dict[group[{}][group_id]] after operation:".format(j), group_id_dict[group[j]["group_id"]])
                    # print("group_id_dict[group[{}][group_id]] after operation:".format(i), group_id_dict[group[i]["group_id"]])
                    # new_group_id = False
                    # TODO 1: Use set or sth to combine two groups to one. DONE
                    # TODO 2: Accelerate! One minute is too slow
                    # TODO 3: Fix the group_id skip problem
            # else:
            #     group[j]["group_id"] = temp_group_id
            search_fail_time = 0
            j += 1
            # initial = False
        else:
            search_fail_time += 1
            j += 1
        if search_fail_time > search_limit or j >= len(group):
            i += 1
            j = i + 1
            if new_group_id: group_id += 1
            new_group_id = False
            search_fail_time = 0
            # initial = True

    return group

######## Source of finding closest distance between two lines: https://stackoverflow.com/questions/2824478/ ########
# But this seems bugged :(

def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):
    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''
    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross)**2
    
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    # if not denom:
    # if denom < 0.1:
    d0 = np.dot(_A,(b0-a0))
    
    # Overlap only possible with clamping
    if clampA0 or clampA1 or clampB0 or clampB1:
        d1 = np.dot(_A,(b1-a0))
        
        # Is segment B before A?
        if d0 <= 0 >= d1:
            if clampA0 and clampB1:
                if np.absolute(d0) < np.absolute(d1):
                    return a0, b0, np.linalg.norm(a0-b0)
                return a0, b1, np.linalg.norm(a0-b1)
            
        # Is segment B after A?
        elif d0 >= magA <= d1:
            if clampA1 and clampB0:
                if np.absolute(d0) < np.absolute(d1):
                    return a1, b0, np.linalg.norm(a1-b0)
                return a1, b1, np.linalg.norm(a1-b1)
            
    # print("Result of parallel line distance:", np.linalg.norm(((d0 * _A) + a0) - b0))
    # Segments overlap, return distance between parallel segments
    return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

def distance_btw_line_and_pt(line_pt0, line_pt1, point):
    '''
    Given one line defined by numpy.array pairs (pt0, pt1)
    Return the perpendicular distance & real distance from the closest point on the segment
    '''
    return np.linalg.norm(np.cross(line_pt1 - line_pt0, point - line_pt0))/np.linalg.norm(line_pt1 - line_pt0), LineString([line_pt0, line_pt1]).distance(Point(point))

def vis_link(background, reports, junc_color=(0, 0, 255), link_color=(0, 127, 0), unit="cm"):
    """
    Visualize links with background.
    Args:
        background (numpy.ndarray): a background image.
        reports (dict): an inspection report with the following format.
            {
                "S0": {"points": [[0, 0], [10, 10]], "length": 12.521},
                "S1": ...
            }
        junc_color (tuple): color of junctions.
        link_color (tuple): color of links.
        unit (str): unit of links.
        
    Returns:
        a numpy array
    """
    background = background.astype(np.uint8)
    for spacing in reports['spacing'].values():
        (x1, y1), (x2, y2) = spacing['points']
        cv2.line(background, (x1, y1), (x2, y2), (0, 0, 0), 7)
        cv2.line(background, (x1, y1), (x2, y2), link_color, 3)
        cv2.circle(background, (x1, y1), 5, junc_color, 2)
        cv2.circle(background, (x2, y2), 5, junc_color, 2)
        if spacing['length'] is not None:
            text = format(spacing['length'], '.2f')
            ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 3)[0]
            cv2.putText(background, text, ((x1+x2)//2-ts[0]//2, (y1+y2)//2+ts[1]//2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 3)
            ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)[0]
            cv2.putText(background, text, ((x1+x2)//2-ts[0]//2, (y1+y2)//2+ts[1]//2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
    
    return background

def vis_group(background, groups, junc_color=(0, 0, 255), link_color=(0, 127, 0), unit="cm"):
    background = background.astype(np.uint8)
    # Visualization        
    # for group in groups.values():
    for gi, group in groups.items():
        for info in group:
            points = info['points']
            mx, my = np.mean(points, axis=0, dtype=int)
            # color = self.colors[1] if info['pass'] else (0, 255, 255)
            # color = link_color if info['pass'] else (0, 255, 255)
            if info['pass']:
                if gi == "horizontal":
                    color = (0, 127, 0)
                elif gi == "vertical":
                    color = (0, 0, 127)
                else:
                    color = (127, 0, 0)
            else:
                color = (0, 255, 255)
            cv2.line(background, points[0], points[1], (0, 0, 0), 7)
            cv2.line(background, points[0], points[1], color, 3)
            cv2.circle(background, points[0], 8, junc_color, -1)
            cv2.circle(background, points[1], 8, junc_color, -1)
            if info['length'] is not None:
                text = '{:.2f}cm'.format(info['length'])
                font = cv2.FONT_HERSHEY_SIMPLEX
                # (text_width, text_height) = cv2.getTextSize(text, font, fontScale=0.5, thickness=3)[0]
                cv2.putText(background, text, (mx-25, my-5), font, 0.5, (0,0,0), 3) #, cv2.LINE_AA)
                # (text_width, text_height) = cv2.getTextSize(text, font, fontScale=0.5, thickness=1)[0]
                # cv2.rectangle(background, (mx-25, my-5), (mx+text_width-25, my-text_height-5), (0,0,0), -1)
                cv2.putText(background, text, (mx-25, my-5), font, 0.5, (255,255,255), 1) #, cv2.LINE_AA)

    return background

def vis_group_rotated(background, groups, junc_color=(0, 0, 255), link_color=(0, 127, 0), unit="cm"):
    background = background.astype(np.uint8)
    # Visualization        
    # for group in groups.values():
    for gi, group in groups.items():
        for info in group:
            # color = self.colors[1] if info['pass'] else (0, 255, 255)
            # color = link_color if info['pass'] else (0, 255, 255)
            # if info['pass']:
            if gi == "horizontal":
                color = (0, 127, 0)
            elif gi == "vertical":
                color = (0, 0, 127)
            else:
                # color = (127, 0, 0)
                continue
            # else:
            #     color = (0, 255, 255)

            points = info['rotated_points_int']
            mx, my = np.mean(points, axis=0, dtype=int)

            pt1 = points[0]
            pt2 = points[1]
            cv2.line(background, pt1, pt2, (0, 0, 0), 7)
            cv2.line(background, pt1, pt2, color, 3)
            cv2.circle(background, pt1, 8, junc_color, -1)
            cv2.circle(background, pt2, 8, junc_color, -1)
            if info['length'] is not None:
                text = '{:.2f}cm'.format(info['length'])
                font = cv2.FONT_HERSHEY_SIMPLEX
                # (text_width, text_height) = cv2.getTextSize(text, font, fontScale=0.5, thickness=3)[0]
                cv2.putText(background, text, (mx-25, my-5), font, 0.5, (0,0,0), 3) #, cv2.LINE_AA)
                # (text_width, text_height) = cv2.getTextSize(text, font, fontScale=0.5, thickness=1)[0]
                # cv2.rectangle(background, (mx-25, my-5), (mx+text_width-25, my-text_height-5), (0,0,0), -1)
                cv2.putText(background, text, (mx-25, my-5), font, 0.5, (255,255,255), 1) #, cv2.LINE_AA)

    return background

def process_group(lines, crosses, length_per_pixel):
    start_time = time.time()
    print("processing group...")
    line_list = []
    for info in lines.values():
        line_list.append(LineString([pt for pts in info for pt in pts["points"]]))
    line_key = list(lines.keys())
    crosses_list = []
    for info in crosses.values():
        crosses_list.append(LineString([pt for pts in info for pt in pts["points"]]))
    line_pointer0 = 0
    line_pointer1 = 1
    report = dict()
    while line_pointer0 < len(lines) - 1:
        has_intersect = False
        base_line = line_list[line_pointer0]
        test_line = line_list[line_pointer1]
        for cross in crosses_list:
            if base_line.intersects(cross) and test_line.intersects(cross):
                intersect0 = base_line.intersection(cross)
                # pt0 = np.asarray(intersect0.coords) if intersect0.geom_type == "Point" else np.asarray(intersect0[0].coords)
                if intersect0.geom_type == "Point":
                    pt0 = np.asarray(intersect0.coords)
                elif intersect0.geom_type == "LineString":
                    print("Warning: intersection is LineString format, which is rare.")
                    print("intersect0:", intersect0)
                    pt0 = np.asarray(intersect0.coords[0])
                else:
                    pt0 = np.asarray(intersect0[0].coords)
                intersect1 = test_line.intersection(cross)
                # pt1 = np.asarray(intersect1.coords) if intersect1.geom_type == "Point" else np.asarray(intersect1[0].coords)
                if intersect1.geom_type == "Point":
                    pt1 = np.asarray(intersect1.coords)
                elif intersect1.geom_type == "LineString":
                    print("Warning: intersection is LineString format, which is rare.")
                    print("intersect1:", intersect1)
                    pt1 = np.asarray(intersect1.coords[0])
                else:
                    pt1 = np.asarray(intersect1[0].coords)
                length = float(np.linalg.norm(pt0 - pt1) * length_per_pixel * 100)
                if not has_intersect:
                    has_intersect = True
                    report_key = line_key[line_pointer0]
                    report[report_key] = {
                        "connect_line_id": line_key[line_pointer1],
                        "lengths": [length]
                    }
                else:
                    report[report_key]["lengths"].append(length)
        if has_intersect:
            length_list = report[report_key]["lengths"]
            report[report_key]["mean"] = np.mean(length_list)
            report[report_key]["median"] = np.median(length_list)
            report[report_key]["std"] = np.std(length_list)
            report[report_key]["min"] = min(length_list)
            report[report_key]["max"] = max(length_list)
            line_pointer0 = line_pointer1
            line_pointer1 += 1
        else:
            line_pointer1 += 1
            if line_pointer1 >= len(lines):
                line_pointer0 += 1
                line_pointer1 = line_pointer0 + 1
    end_time = time.time()
    print('Time for processing group:', end_time-start_time, 'seconds')
    return report

def sort_by_groupid(info, direction):
    start_time = time.time()
    print("sorting group id...")
    # Sorting orientation
    group_id_set = set([IF['group_id'] for IF in info])
    print("set of group_id:", group_id_set)
    group_id_set.discard(-1)
    # print("set of group_id after discarding -1:", group_id_set)
    # groups = {i: [] for i in range(max([IF['group_id'] for IF in info]) + 1)}
    groups = {i: [] for i in group_id_set}
    # groups = {"horizontal": [], "vertical": [], "outlier": []}
    for IF in info: #.values():
        if IF['group_id'] >= 0:
            groups[IF['group_id']].append({
                'points': IF['points'], 'length': IF['length'], 'orientation': IF['orientation'], 'location': IF['location'], 
                'rotated_points': IF['rotated_points'], 'rotated_points_int': IF['rotated_points_int'], 'pass': IF['pass'],
                })

    group_id_to_discard = set()
    # for i, info in groups.items():
        # print("i:", i)

    for gi, group in groups.items():
        if direction == "horizontal":
            groups[gi] = sorted(group, key = lambda k: k['location'][0])
        elif direction == "vertical":
            groups[gi] = sorted(group, key = lambda k: k['location'][1])

        if len(group) < 3:
            # groups.pop(i)
            group_id_to_discard.add(gi)
            print("{} discarded".format(gi))
    
    for id in group_id_to_discard:
        groups.pop(id)

    # print("groups after sort_by_groupid:", groups)
    end_time = time.time()
    print("Time for sorting:", end_time-start_time, "seconds")
    return groups

def vis_link_group(background, group, args, filename, direction, junc_color=(0, 0, 255), link_color=(0, 127, 0), unit="cm"):
    start_time = time.time()
    print("visualizing link group...")
    background = background.astype(np.uint8)

    # plt.clf()
    # plt.cla()
    plt.close()
    plt.figure(figsize=(background.shape[1]/100, background.shape[0]/100))
    plt.axis('off')
    plt.xlim([0, background.shape[1]])
    plt.ylim([background.shape[0], 0])    # Invert y-axis to fit the y direction in openCV

    for i, info in tqdm(group.items()):
        line = LineString([pt for pts in info for pt in pts["points"]])
        x, y = line.xy
        plt.plot(x, y, linewidth=3)
        plt.text(x[0], y[0], str(i))

    plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    # plt.title("{} : linegroup".format(direction))
    
    plt.savefig(os.path.join(args.output_dir, filename + '-linegroup-{}.jpg'.format(direction)), bbox_inches='tight', pad_inches = 0)

    end_time = time.time()
    print('Time for visualizing line group:', end_time-start_time, 'seconds')
    return group

def vis_group_spacing(background, group, report, args, filename, direction, junc_color=(0, 0, 255), link_color=(0, 127, 0), unit="cm"):
    start_time = time.time()
    print("Visualizing group of spacings...")
    background = background.astype(np.uint8)

    # NOTE: Continue from vis_link_group(). Notice the order!! Or pyplot may behave unexpectedly.
    # plt.clf()
    # plt.cla()
    # plt.figure(figsize=(40, 40))
    # plt.xlim([0, background.shape[1]])
    # plt.ylim([0, background.shape[0]])

    for i, info in tqdm(report.items()):
        # print([pt for pts in info for pt in pts["points"]])
        # line = LineString([pt for pts in info for pt in pts["points"]])
        # x, y = line.xy
        # plt.plot(x, y, linewidth=3)
        x0, y0 = group[i][0]["location"]
        x1, y1 = group[info["connect_line_id"]][0]["location"]
        plt.annotate("", xy=(x0, y0), xytext=(x1, y1), arrowprops=dict(connectionstyle="arc3, rad=0.1", arrowstyle="<|-|>"))
        color = "r" if info["outliers"] else "w"
        if direction == "horizontal":
            plt.text((x0+x1)/2, (y0+y1)/2, "{:.2f} +- {:.2f} cm".format(info["mean"], info["std"]), fontsize="large", verticalalignment="center", horizontalalignment="center", bbox=dict(boxstyle="Round", ec="None", fc=color, alpha=0.8))
        else:
            plt.text((x0+x1)/2, (y0+y1)/2, "{:.2f}\n+-{:.2f}cm".format(info["mean"], info["std"]), fontsize="large", verticalalignment="center", horizontalalignment="center", bbox=dict(boxstyle="Round", ec="None", fc=color, alpha=0.8))

    # plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    # plt.imshow(np.flip(background, axis=0))    # I use np.flip because y's direction in array image is up to down, while that in pyplot plot line is down to up    # <- forget that = =
    # plt.title("{} : group of spacing".format(direction))
    plt.savefig(os.path.join(args.output_dir, filename + '-spacing_group-{}.jpg'.format(direction)), bbox_inches='tight', pad_inches = 0)


    end_time = time.time()
    print('Time for visualizing group of spacing:', end_time-start_time, 'seconds')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="inference images & coord folder dir", default="./stitched_image")
    parser.add_argument("-o", "--output_dir", help="where to save the inference results", default="inference_results")
    parser.add_argument("-p", "--overlap_pixel", type=float, help="How much to overlap when slicing images into small ones. 0<x<1 provides fraction of images, while int x > 1 provides exact pixel.", default=280)
    parser.add_argument("-s", "--scale", help="Scale of window of sliced images", type=float, default=1)
    parser.add_argument("--min_num_for_active_inspection", help="The minimun number of detected spacings with the same orientation", type=int, default=5)
    parser.add_argument("--tolerance_bias", help="The tolerance bias between the detected spacings and pseudo-ground truth spacings (unit: cm)", type=float, default=5)
    parser.add_argument("--tolerance_ratio", help="The tolerance ratio of pseudo-ground truth spacings", type=float, default=0.2)
    parser.add_argument("-v", "--verbose", help="Whether to show the intermediate inference results", action="store_true", default=False)
    parser.add_argument("-m", "--model_dir", help="mask r-cnn model dir", default="/home/r09521612/model_training/0803_rgb/model_final.pth")
    args = parser.parse_args()
    main(args)
