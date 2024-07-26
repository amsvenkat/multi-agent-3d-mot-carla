import datetime
import logging
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import wandb

from smoke.utils.metric_logger import MetricLogger
from smoke.utils.comm import get_world_size


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
    project="object-det-model-v3",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": cfg.SOLVER.BASE_LR,
    "architecture": "DLA34-SMOKE",
    "dataset": "CARLA",
    "epochs":  cfg.SOLVER.MAX_ITERATION,
    "optimizer": cfg.SOLVER.OPTIMIZER,
    "batch_size": cfg.SOLVER.IMS_PER_BATCH,
    'loss': 'hm : FocalLoss, dim: L2norm',
    "depth_ref" : cfg.MODEL.SMOKE_HEAD.DEPTH_REFERENCE ,
    "dim_ref" : cfg.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE,
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

    wandb.watch(model, log="all")
    for epoch in range(epoch_total):
        
        epoch = epoch + 1
        logger.info("Epoch: {}".format(epoch))
        
        model.train()
        torch.cuda.empty_cache()
        
        for i, data in enumerate(data_loader):
            
            if i % 10 == 0:
                logger.info("In Iteration {}".format(i))

            data_time = time.time() - end

            arguments["iteration"] = epoch
           
            images = data["images"].to(device)

            targets = [target.to(device) for target in data["targets"]]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
                        
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

            if i % 10 == 0 or i ==0:
                logger.info(
                    meters.delimiter.join(
                        [
                        #    "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.8f}",
                            "max men: {memory:.0f}",
                        ]
                    ).format(
                    #    eta=eta_string,
                        iter=i,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                    )
                )


        del images, targets

        #Validation 
        val_losses_reduced = 0.0

        with torch.no_grad():
            
            model.eval()
            logger.info("Validation")
            
            for i, sample in enumerate(data_loader_val):
                
                if i % 10 == 0:
                    logger.info("In Iteration {}".format(i))
                
                images = sample["images"].to(device)
                targets = [target.to(device) for target in sample["targets"]]

                val_loss_dict = model(images, targets)
                val_losses = sum(loss for loss in val_loss_dict.values())
                
                # rduce losses over all GPUs for logging purposes
                val_loss_dict_reduced = reduce_loss_dict(val_loss_dict)
                val_losses_reduced = sum(loss for loss in val_loss_dict_reduced.values())
                meters.update(val_loss=val_losses_reduced, **val_loss_dict_reduced)
        
        del images, targets
        
        epoch_time = time.time()- end
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
        
        if best_test_loss > val_losses_reduced:
            best_test_loss = val_losses_reduced
            checkpointer.save("model_best_{:03d}".format(epoch), **arguments)  
               
        # fixme: do we need checkpoint_period here
        if epoch in cfg.SOLVER.STEPS:
            checkpointer.save("model_{:03d}".format(epoch), **arguments)
       
        if epoch % 10 == 0:
            checkpointer.save("model_intermediate_{:03d}".format(epoch), **arguments)

        if epoch == epoch_total :
            checkpointer.save("model_final_{:03d}".format(epoch), **arguments)
        
        wandb.log({
                        "total_loss" :losses_reduced,
                        "individual_loss_heatmap" : loss_dict_reduced['hm_loss'],
                        "individual_loss_reg" :loss_dict_reduced['reg_loss'],
                        "total_val_loss" : val_losses_reduced,
                        "individual_val_loss_heatmap" : val_loss_dict_reduced['hm_loss'],
                        "individual_val_loss_reg" : val_loss_dict_reduced['reg_loss'],
                        }) 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (epoch)
        )
    )
    wandb.finish()