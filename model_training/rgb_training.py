import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_train_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

import detectron2.utils.comm as comm
import torch, torchvision

import numpy as np

from detectron2.utils.logger import setup_logger
import logging
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import CommonMetricPrinter, EventStorage, JSONWriter #, TensorboardXWriter

from detectron2.engine import launch


def main():
    ###
    register_coco_instances("batch_train", {}, "./0310_label/anno/annotations.json", "./0310_label/rgb")
    register_coco_instances("batch_val", {}, "./0310_label/anno/annotations.json", "./0310_label/rgb")

    setup_logger()
    logger = logging.getLogger("detectron2")

    cfg_source = get_cfg()
    cfg_source.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    # cfg_source.merge_from_file(model_zoo.get_config_file("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py"))    # Detectron2's new MRCNN baseline, to be investigated
    # cfg_source.DATASETS.TRAIN = ("2019_train",)
    cfg_source.DATASETS.TRAIN = ("batch_train",)
    # cfg_source.DATASETS.TEST = ()
    # cfg_source.DATASETS.VAL = ("2019_val",)
    cfg_source.DATASETS.VAL = ("batch_val",)
    # cfg_source.DATASETS.TEST = ("2021_test",)
    cfg_source.DATASETS.TEST = ("batch_val",)
    # cfg_source.TEST.EVAL_PERIOD = 300
    cfg_source.DATALOADER.NUM_WORKERS = 2
    cfg_source.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    # cfg_source.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py")
    cfg_source.SOLVER.IMS_PER_BATCH = 4
    cfg_source.SOLVER.BASE_LR = 0.00003
    cfg_source.SOLVER.WEIGHT_DECAY = 0.001
    cfg_source.SOLVER.MAX_ITER = 50000
    cfg_source.SOLVER.STEPS = (300,)
    cfg_source.SOLVER.GAMMA = 0.5
    cfg_source.INPUT.MIN_SIZE_TRAIN = (0,)
    cfg_source.INPUT.MIN_SIZE_TEST = 0
    cfg_source.OUTPUT_DIR = './0310_rgb'
    os.makedirs(cfg_source.OUTPUT_DIR, exist_ok=True)
    cfg_source.MODEL.ROI_HEADS.NUM_CLASSES = 2

    cfg_source.SOLVER.CHECKPOINT_PERIOD = 2000
    
    ###
    cfg_source.MODEL.WEIGHTS = "./pretrain_contrastive_rcb.pth"     # Import pretrained weight from previous Steelscape models

    model = build_model(cfg_source)

    # cfg_target = cfg_source.clone()
    # cfg_target.DATASETS.TRAIN = cfg_source.DATASETS.TEST

    cfg_source_val = cfg_source.clone()
    cfg_source_val.DATASETS.TRAIN = cfg_source.DATASETS.VAL

    """ TRAIN """
    resume = False

    model.train()
    
    print(model)
    
    optimizer = build_optimizer(cfg_source, model)
    scheduler = build_lr_scheduler(cfg_source, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg_source.OUTPUT_DIR, optimizer = optimizer, scheduler = scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg_source.MODEL.WEIGHTS, resume = resume).get("iteration", -1) + 1
    )
    max_iter = cfg_source.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg_source.SOLVER.CHECKPOINT_PERIOD, max_iter = max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg_source.OUTPUT_DIR, "metrics.json")),
            # TensorboardXWriter(cfg_source.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    
    # i = 1
    # current_epoch = 0
    # data_len = 176
    # max_epoch = cfg_source.SOLVER.MAX_ITER / data_len # max iter / min(data_len(data_source, data_target))

    # alpha3 = 0
    # alpha4 = 0
    # alpha5 = 0

    # da_ratio = 100    # ratio of Label predictor : domain classifier

    # data_loader_source = build_detection_train_loader(cfg_source)
    # data_loader_target = build_detection_train_loader(cfg_target)
    # data_loader_source_val = build_detection_train_loader(cfg_source_val)
    # data_loader_da_source = build_detection_train_loader(cfg_da_source)
    # data_loader_da_target = build_detection_train_loader(cfg_da_target)
    # data_loader_da_source_val = build_detection_train_loader(cfg_da_source_val)
    # data_loader_da_target_val = build_detection_train_loader(cfg_da_target_val)

    data_loader_source = build_detection_train_loader(cfg_source)
    data_loader_source_val = build_detection_train_loader(cfg_source_val)

    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        # for data_source, data_target, data_source_val, data_da_source, data_da_target, data_da_source_val, data_da_target_val, iteration in zip(data_loader_source, data_loader_target, data_loader_source_val, data_loader_da_source, data_loader_da_target, data_loader_da_source_val, data_loader_da_target_val, range(start_iter, max_iter)):
        for data_source, data_source_val, iteration in zip(data_loader_source, data_loader_source_val, range(start_iter, max_iter)):
            # iteration = iteration + 1
            storage.step()
            
            # if (iteration % data_len) == 0:
            #     current_epoch += 1
            #     i = 1

            # p = float( i + current_epoch * data_len) / max_epoch / data_len
            # alpha = 2. / ( 1. + np.exp( -10 * p)) - 1
            # i += 1

            # alpha3 = alpha
            # alpha4 = alpha
            # alpha5 = alpha

            # if alpha3 > 0.5:
            #     alpha3 = 0.5

            # if alpha4 > 0.5:
            #     alpha4 = 0.5

            # if alpha5 > 0.1:
            #     alpha5 = 0.1

            # da_bool = iteration % da_ratio == 0
            
            loss_dict = model(data_source)

            # if da_bool:
            #     loss_dict_source = model(data_da_source, False, True, alpha3, alpha4, alpha5)
            #     loss_dict_target = model(data_da_target, True, True, alpha3, alpha4, alpha5)
            #     loss_dict_source["loss_r3"] += loss_dict_target["loss_r3"]
            #     loss_dict_source["loss_r4"] += loss_dict_target["loss_r4"]
            #     loss_dict_source["loss_r5"] += loss_dict_target["loss_r5"]

            #     loss_dict_source["loss_r3"] *= 0.5
            #     loss_dict_source["loss_r4"] *= 0.5
            #     loss_dict_source["loss_r5"] *= 0.5

            #     loss_dict.update(loss_dict_source)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            # if da_bool:
            #     losses_reduced = sum(loss for loss in loss_dict_reduced.values()) - 2 * (loss_dict_reduced["loss_r3"] + loss_dict_reduced["loss_r4"] + loss_dict_reduced["loss_r5"])
            #     da_losses_reduced = (loss_dict_reduced["loss_r3"] + loss_dict_reduced["loss_r4"] + loss_dict_reduced["loss_r5"])
            #     if comm.is_main_process():
            #         storage.put_scalars(feature_extractor_loss=losses_reduced, da_loss=da_losses_reduced, **loss_dict_reduced)
            # else:
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)


            """ VALIDATION """
            with torch.no_grad():
                loss_dict = model(data_source_val)
                
                losses_val = sum(loss_dict.values())
                assert torch.isfinite(losses_val).all(), loss_dict

                loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                    comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_val_loss=losses_reduced, **loss_dict_reduced)

                # loss_dict = model(data_target)
                
                # losses_test = sum(loss_dict.values())
                # assert torch.isfinite(losses_test).all(), loss_dict

                # loss_dict_reduced = {"test_" + k: v.item() for k, v in 
                #                     comm.reduce_dict(loss_dict).items()}
                # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                # if comm.is_main_process():
                #     storage.put_scalars(feature_extractor_test_loss=losses_reduced, **loss_dict_reduced)
                
                # if da_bool:
                #     loss_dict_source = model(data_da_source_val, False, True, alpha3, alpha4, alpha5)
                #     loss_dict_target = model(data_da_target_val, True, True, alpha3, alpha4, alpha5)
                #     loss_dict_source["loss_r3"] += loss_dict_target["loss_r3"]
                #     loss_dict_source["loss_r4"] += loss_dict_target["loss_r4"]
                #     loss_dict_source["loss_r5"] += loss_dict_target["loss_r5"]

                #     loss_dict_source["loss_r3"] *= 0.5
                #     loss_dict_source["loss_r4"] *= 0.5
                #     loss_dict_source["loss_r5"] *= 0.5

                #     losses_val = sum(loss_dict_source.values())
                #     assert torch.isfinite(losses_val).all(), loss_dict_source

                #     loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                #                         comm.reduce_dict(loss_dict_source).items()}
                #     losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                #     if comm.is_main_process():
                #         storage.put_scalars(da_val_loss=losses_reduced, **loss_dict_reduced)


            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            # storage.put_scalar("alpha", alpha, smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and ((iteration % 20 == 0) or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


if __name__ == "__main__":
    # launch(main, 2)    # Use more GPUs to prevent CUDA out of memory
    main()
