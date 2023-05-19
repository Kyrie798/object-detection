import os
import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data import DataLoader

from utils.utils import get_anchors, get_classes
from model.yolo import YoloBody
from utils.dataset import YoloDataset, yolo_dataset_collate
from utils.history import LossHistory, EvalCallback
from utils.utils_fit import fit_one_epoch
from model.yolo_training import get_lr_scheduler, set_optimizer_lr, YOLOLoss

if __name__ == "__main__":
    fp16 = True # 混合精度训练

    classes_path = "model_data/voc_classes.txt"
    anchors_path = "model_data/yolo_anchors.txt"
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # 如果设置了model_path， 则pretrained无效；
    # 如果没有设置model_path，pretrained = True，导入backbone的预训练权重
    pretrained = False
    model_path = "model_data/yolo_weights.pth"
    input_shape = [416, 416]

    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 16

    UnFreeze_Epoch = 30 # 总Epoch
    Unfreeze_batch_size = 8

    Freeze_Train = True

    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01

    save_period = 10 # 每10个epcoh保存一次weights
    save_dir = "logs"

    train_annotation_path = "2012_train.txt"
    val_annotation_path = "2012_val.txt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    model = YoloBody(anchors_mask, num_classes, pretrained=pretrained)

    # 导入预训练权重
    if model_path != "":
        print("Load weights {}.".format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    print("Successful Load Key:{}. \nSuccessful Load Key Num:{}".format(str(load_key)[:500], len(load_key)))
    print("Fail Load Key:{}. \nFail Load Key Num:{}".format(str(no_load_key)[:500], len(no_load_key)))
    
    # 记录loss
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, anchors_mask)
    time_str = datetime.datetime.strftime(datetime.datetime.now(),"%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    # 混合精度训练
    if fp16:
        scaler = GradScaler()

    model_train = model.train()
    model_train = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_train = model_train.cuda()

    # 读取数据集路径
    with open(train_annotation_path, "r") as f:
        train_lines = f.readlines()
    with open(val_annotation_path, "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    
    # 冻住主干网络
    UnFreeze_flag = False
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    nbs = 64
    lr_limit_max = 5e-2
    lr_limit_min = 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # 定义optimizer
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)    
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)    
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight) 
    optimizer = optim.SGD(pg0, Init_lr_fit, momentum=0.937, nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": 5e-4})
    optimizer.add_param_group({"params": pg2})

    # 学习率下降公式
    lr_scheduler_func = get_lr_scheduler(Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    train_dataset = YoloDataset(train_lines, input_shape, num_classes, train=True)
    val_dataset = YoloDataset(val_lines, input_shape, num_classes, train=False)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0, 
                     pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, sampler=None)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=0, 
                         pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, sampler=None)
    
    # 记录eval的map曲线
    eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, 
                                 val_lines, log_dir, True, eval_flag=True, period=10)
    
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # 如果当前epoch>冻住的epoch，则解冻
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size
            nbs = 64
            lr_limit_max = 5e-2
            lr_limit_min = 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            lr_scheduler_func = get_lr_scheduler(Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=None)
            val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True, 
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=None)
            
            UnFreeze_flag = True
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, 
                      train_dataloader, val_dataloader, UnFreeze_Epoch, device, fp16, scaler, save_period, save_dir)
    loss_history.writer.close()
