import os

import torch
from nets.pspnet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score, mIoU

import wandb


def _segmentation_loss(outputs, pngs, labels, weights, num_classes, dice_loss, focal_loss):
    if focal_loss:
        loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
    else:
        loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

    if dice_loss:
        loss = loss + Dice_loss(outputs, labels)

    return loss


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0, weight_save_dir=None,
                  multi_task=False, object_num_classes=None, shoreline_num_classes=None, object_cls_weights=None, shoreline_cls_weights=None, shoreline_loss_weight=1.0):
    total_loss      = 0
    total_f_score   = 0
    total_miou      = 0
    total_object_miou = 0
    total_shoreline_miou = 0

    val_loss        = 0
    val_f_score     = 0
    val_miou        = 0
    val_object_miou = 0
    val_shoreline_miou = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        if multi_task:
            imgs, object_pngs, object_labels, shoreline_pngs, shoreline_labels = batch
        else:
            imgs, pngs, labels = batch
        with torch.no_grad():
            if multi_task:
                object_weights = torch.from_numpy(object_cls_weights)
                shoreline_weights = torch.from_numpy(shoreline_cls_weights)
            else:
                weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                if multi_task:
                    object_pngs = object_pngs.cuda(local_rank)
                    object_labels = object_labels.cuda(local_rank)
                    shoreline_pngs = shoreline_pngs.cuda(local_rank)
                    shoreline_labels = shoreline_labels.cuda(local_rank)
                    object_weights = object_weights.cuda(local_rank)
                    shoreline_weights = shoreline_weights.cuda(local_rank)
                else:
                    pngs    = pngs.cuda(local_rank)
                    labels  = labels.cuda(local_rank)
                    weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            if multi_task:
                object_outputs, shoreline_outputs = model_train(imgs)
                object_loss = _segmentation_loss(
                    object_outputs, object_pngs, object_labels, object_weights,
                    object_num_classes, dice_loss, focal_loss
                )
                shoreline_loss = _segmentation_loss(
                    shoreline_outputs, shoreline_pngs, shoreline_labels, shoreline_weights,
                    shoreline_num_classes, dice_loss, focal_loss
                )
                loss = object_loss + shoreline_loss_weight * shoreline_loss
                outputs = object_outputs
                labels = object_labels
            else:
                outputs = model_train(imgs)
                loss = _segmentation_loss(outputs, pngs, labels, weights, num_classes, dice_loss, focal_loss)

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                # _f_score = f_score(outputs, labels)
                if multi_task:
                    train_object_miou = mIoU(object_outputs, object_labels)
                    train_shoreline_miou = mIoU(shoreline_outputs, shoreline_labels)
                    train_miou = (train_object_miou + train_shoreline_miou) / 2
                else:
                    train_miou = mIoU(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                if multi_task:
                    object_outputs, shoreline_outputs = model_train(imgs)
                    object_loss = _segmentation_loss(
                        object_outputs, object_pngs, object_labels, object_weights,
                        object_num_classes, dice_loss, focal_loss
                    )
                    shoreline_loss = _segmentation_loss(
                        shoreline_outputs, shoreline_pngs, shoreline_labels, shoreline_weights,
                        shoreline_num_classes, dice_loss, focal_loss
                    )
                    loss = object_loss + shoreline_loss_weight * shoreline_loss
                    outputs = object_outputs
                    labels = object_labels
                else:
                    outputs = model_train(imgs)
                    loss = _segmentation_loss(outputs, pngs, labels, weights, num_classes, dice_loss, focal_loss)

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    # _f_score = f_score(outputs, labels)
                    if multi_task:
                        train_object_miou = mIoU(object_outputs, object_labels)
                        train_shoreline_miou = mIoU(shoreline_outputs, shoreline_labels)
                        train_miou = (train_object_miou + train_shoreline_miou) / 2
                    else:
                        train_miou = mIoU(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        # total_f_score   += _f_score.item()
        total_miou      += train_miou.item()
        if multi_task:
            total_object_miou += train_object_miou.item()
            total_shoreline_miou += train_shoreline_miou.item()
        
        if local_rank == 0:
            postfix = {'total_loss': total_loss / (iteration + 1),
                       'mIoU': total_miou / (iteration + 1),
                       'lr': get_lr(optimizer)}
            if multi_task:
                postfix.update({
                    'obj_mIoU': total_object_miou / (iteration + 1),
                    'shore_mIoU': total_shoreline_miou / (iteration + 1),
                })
            pbar.set_postfix(**postfix)
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        if multi_task:
            imgs, object_pngs, object_labels, shoreline_pngs, shoreline_labels = batch
        else:
            imgs, pngs, labels = batch
        with torch.no_grad():
            if multi_task:
                object_weights = torch.from_numpy(object_cls_weights)
                shoreline_weights = torch.from_numpy(shoreline_cls_weights)
            else:
                weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                if multi_task:
                    object_pngs = object_pngs.cuda(local_rank)
                    object_labels = object_labels.cuda(local_rank)
                    shoreline_pngs = shoreline_pngs.cuda(local_rank)
                    shoreline_labels = shoreline_labels.cuda(local_rank)
                    object_weights = object_weights.cuda(local_rank)
                    shoreline_weights = shoreline_weights.cuda(local_rank)
                else:
                    pngs    = pngs.cuda(local_rank)
                    labels  = labels.cuda(local_rank)
                    weights = weights.cuda(local_rank)

            #----------------------#
            #   前向传播
            #----------------------#
            if multi_task:
                object_outputs, shoreline_outputs = model_train(imgs)
                object_loss = _segmentation_loss(
                    object_outputs, object_pngs, object_labels, object_weights,
                    object_num_classes, dice_loss, focal_loss
                )
                shoreline_loss = _segmentation_loss(
                    shoreline_outputs, shoreline_pngs, shoreline_labels, shoreline_weights,
                    shoreline_num_classes, dice_loss, focal_loss
                )
                loss = object_loss + shoreline_loss_weight * shoreline_loss
                outputs = object_outputs
                labels = object_labels
            else:
                outputs = model_train(imgs)
                loss = _segmentation_loss(outputs, pngs, labels, weights, num_classes, dice_loss, focal_loss)
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            # _f_score    = f_score(outputs, labels)
            if multi_task:
                object_miou = mIoU(object_outputs, object_labels)
                shoreline_miou = mIoU(shoreline_outputs, shoreline_labels)
                _miou = (object_miou + shoreline_miou) / 2
            else:
                _miou = mIoU(outputs, labels)

            val_loss    += loss.item()
            # val_f_score += _f_score.item()
            val_miou    += _miou.item()
            if multi_task:
                val_object_miou += object_miou.item()
                val_shoreline_miou += shoreline_miou.item()
            
        if local_rank == 0:
            postfix = {'val_loss': val_loss / (iteration + 1),
                       'mIoU': val_miou / (iteration + 1),
                       'lr': get_lr(optimizer)}
            if multi_task:
                postfix.update({
                    'obj_mIoU': val_object_miou / (iteration + 1),
                    'shore_mIoU': val_shoreline_miou / (iteration + 1),
                })
            pbar.set_postfix(**postfix)
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_miou(val_miou / epoch_step_val)
        # loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
        # eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        log_dict = {
            'epoch': epoch,
            'se seg loss': total_loss / epoch_step,
            'mIoU wl(train)': total_miou / epoch_step,
            'lr': get_lr(optimizer),
            'se seg val_loss': val_loss / epoch_step_val,
            'mIoU wl(eval)': val_miou / epoch_step_val,
        }
        if multi_task:
            log_dict.update({
                'mIoU object(train)': total_object_miou / epoch_step,
                'mIoU shoreline(train)': total_shoreline_miou / epoch_step,
                'mIoU object(eval)': val_object_miou / epoch_step_val,
                'mIoU shoreline(eval)': val_shoreline_miou / epoch_step_val,
            })
        wandb.log(log_dict)
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if len(loss_history.miou) <= 1 or (val_miou / epoch_step_val) >= max(loss_history.miou):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(weight_save_dir, "best_epoch_weights_ep%03d_mIoU%.3f.pth" % (
                    epoch + 1, val_miou / epoch_step_val)))

        # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        #     torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        # if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        #     print('Save best model to best_epoch_weights.pth')
        #     torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        # torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   损失计算
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   损失计算
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f.pth'%((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
