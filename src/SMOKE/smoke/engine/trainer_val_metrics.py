import datetime
import logging
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import wandb
from collections import defaultdict
from smoke.utils.metric_logger import MetricLogger
from smoke.utils.comm import get_world_size

import sys
# sys.path.insert(0, "/export/amsvenkat/project/ma_perception/SMOKE/tools/")

ID_TYPE_CONVERSION = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian'
}


from evaluation.eval import Evaluate

def reduce_metrics(map_epoch, error_list):
    world_size = get_world_size()
    
    if world_size < 2:
        return map_epoch, error_list
    
    with torch.no_grad():
        map_epoch = torch.tensor(map_epoch).to("cuda")
        dist.reduce(map_epoch, dst=0)
        for k,v in error_list.items():
            error_list[k] = torch.tensor(v).to("cuda")
            dist.reduce(error_list[k], dst=0)
        
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            map_epoch /= world_size
            error_list = {k:v/world_size for k,v in error_list.items()}

    return map_epoch, error_list

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def get_gts(data_loader_val):
    gts = defaultdict(list)
    for i, sample in enumerate(data_loader_val):
        gts_path = "/export/amsvenkat/project/ma_perception/data/train_v3/label_2/"
        ind = int(sample["img_ids"][0])
        with open(gts_path +str(ind) + ".txt", 'r') as f:
                list_items = []
                lines = f.readlines()
                for line in lines:
                    line = line.split(' ')
                    line = [ float(i) if i not in ['Car', 'Pedestrian'] else str(i)  for i in line ]
                    dictionary = [{'sample_frame': ind ,'class_name': line[0], 'dimension': [line[8], line[9], line[10]],
                                  'location_cam': [line[11], line[12], line[13]], 'rotation': line[14], 'score': line[15]}]
                    list_items.extend(dictionary)
        gts[ind].extend(list_items)
    
    return gts


def do_train(
        cfg,
        distributed,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
):

    wandb.init(
        # set the wandb project where this run will be logged
        project="object-det-model-v6",

        # track hyperparameters and run metadata
        config={
            "learning_rate": cfg.SOLVER.BASE_LR,
            "architecture": "DLA34-SMOKE",
            "dataset": "CARLA",
            "epochs":  cfg.SOLVER.MAX_ITERATION,
            "optimizer": cfg.SOLVER.OPTIMIZER,
            "batch_size": cfg.SOLVER.IMS_PER_BATCH,
            'loss': 'hm : FocalLoss, dim: L2norm',
            "depth_ref": cfg.MODEL.SMOKE_HEAD.DEPTH_REFERENCE,
            "dim_ref": cfg.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE,
        }
    )

    logger = logging.getLogger("smoke.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    best_test_loss = np.inf
    epoch_total = cfg.SOLVER.MAX_ITERATION
    #mAP_epoch = mtranserr_epoch = mscaleerr_epoch = morienterr_epoch = 0
    
    wandb.watch(model, log="all")

    gts = get_gts(data_loader_val)
    best_mAP =0.0
    
    for epoch in range(epoch_total):

        epoch = epoch + 1
        logger.info("Epoch: {}".format(epoch))
        model.train()
        torch.cuda.empty_cache()
        mAP_epoch = 0
        merror_epoch  = {'trans_err': 0, 'scale_err': 0, 'orient_err': 0}

        for i, data in enumerate(data_loader):
            model.train()
            if i % 10 == 0:
                logger.info("In Iteration {}".format(i))
            data_time = time.time() - end
            arguments["iteration"] = epoch
            images = data["images"].to(device)
            targets = [target.to(device) for target in data["targets"]]
            _, loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0 or i == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.8f}",
                            "max men: {memory:.0f}",
                        ]
                    ).format(
                        iter=i,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                    )
                )
            
        scheduler.step()
        del images, targets
  
        with torch.no_grad():
            model.eval()
            logger.info("Validation Metrics")
            preds = {}
            for i, sample in enumerate(data_loader_val):
                images = sample["images"].to(device)
                targets = [target.to(device) for target in sample["targets"]]
                ind = int(sample["img_ids"][0])
                result, _ = model(images, targets)
                result = result.to('cpu')
                list_items = []
                #print(result)
                for result in result:
                    dictionary = [{'sample_frame': ind, 'class_name': ID_TYPE_CONVERSION[int(result[0])], 'dimension': [result[6], result[7], result[8]],
                                   'location_cam': [result[9], result[10], result[11]], 'rotation': result[12], 'score': result[13]}]
                    list_items.extend(dictionary)
                preds[ind] = list_items
            
            #if all(preds.values()) != 0:
            dataset_split = "val"
            eval = Evaluate(dataset_split)
            mAP, error = eval.main_train(preds, gts)
            mAP_epoch, merror_epoch = reduce_metrics(mAP, error)
                #mAP = torch.
            # else:
            #     mAP_epoch = 0
            #     merror_epoch = { k : 1 for k in merror_epoch.keys()}
            #     mAP_epoch, merror_epoch = reduce_metrics(mAP, error)

        #mAP_epoch, merror_epoch = reduce_metrics(mAP, error)
        
        epoch_time = time.time() - end
        end = time.time()
        meters.update(epoch_time=epoch_time)
        eta_seconds = meters.epoch_time.global_avg * (epoch_total - epoch)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        logger.info(
            meters.delimiter.join(
                [
                    "eta: {eta}",
                    "{meters}",
                    "lr: {lr:.8f}",
                    "max men: {memory:.0f}",
                ]
            ).format(
                eta=eta_string,
                meters=str(meters),
                lr=optimizer.param_groups[0]["lr"],
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            )
        )

        if  mAP_epoch > best_mAP :
            best_mAP = mAP_epoch
            checkpointer.save("model_best_metric_{:03d}".format(epoch), **arguments)
        
        # if best_test_loss > val_losses_reduced:
        #     best_test_loss = val_losses_reduced
        #     checkpointer.save("model_best_{:03d}".format(epoch), **arguments)

        if epoch % 10 == 0:
            checkpointer.save(
                "model_intermediate_{:03d}".format(epoch), **arguments)

        if epoch == epoch_total:
            checkpointer.save("model_final_{:03d}".format(epoch), **arguments)
        print("Here")
        wandb.log({
                "X": epoch,
            "total_loss": losses_reduced,
            "individual_loss_heatmap": loss_dict_reduced['hm_loss'],
            "individual_loss_reg": loss_dict_reduced['reg_loss'],
            "mAP": mAP_epoch,
            "trans_error": merror_epoch['trans_err'],
            "orient_error": merror_epoch['orient_err'],
            "scale_error": merror_epoch['scale_err'],}
            )

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (epoch)
        )
    )
    wandb.finish()
