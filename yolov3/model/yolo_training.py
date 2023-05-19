import math
import torch
import numpy as np
import torch.nn as nn

from functools import partial

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio = 1 * (num_classes / 80)

        self.ignore_threshold = 0.5

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def BCELoss(self, pred, target):
        """BCELoss"""
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output
    
    def box_giou(self, b1, b2):
        """
        计算box的giou
        GIoU=IoU-(C-AUB)/C, C是能包裹两个框的最小框
        """
        # b1, b2:[bs, feat_w, feat_h, anchor_num, 4], 4: xywh
        # 求出预测框左上角右下角
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2: 4]
        b1_wh_half = b1_wh/2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        # 求出真实框左上角右下角
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2: 4]
        b2_wh_half = b2_wh/2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # 求真实框和预测框所有的iou
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area

        # 找到包裹两个框的最小框的左上角和右下角
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))

        # 计算对角线距离
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou = iou - (enclose_area - union_area) / enclose_area
        return giou

    def calculate_iou(self, box_a, box_b):
        """计算IoU"""
        # 计算真实框的左上角和右下角    
        b1_x1, b1_x2 = box_a[:, 0] - box_a[:, 2] / 2, box_a[:, 0] + box_a[:, 2] / 2
        b1_y1, b1_y2 = box_a[:, 1] - box_a[:, 3] / 2, box_a[:, 1] + box_a[:, 3] / 2 

        #   计算先验框获得的预测框的左上角和右下角
        b2_x1, b2_x2 = box_b[:, 0] - box_b[:, 2] / 2, box_b[:, 0] + box_b[:, 2] / 2
        b2_y1, b2_y2 = box_b[:, 1] - box_b[:, 3] / 2, box_b[:, 1] + box_b[:, 3] / 2

        # 将真实框和预测框都转化成左上角右下角的形式
        box_a = torch.zeros_like(box_a)
        box_b = torch.zeros_like(box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        # A为真实框的数量，B为先验框的数量
        A = box_a.size(0)
        B = box_b.size(0)

        # 计算交的面积
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        
        # 计算预测框和真实框各自的面积
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        
        # 求IOU
        union = area_a + area_b - inter
        return inter / union
    
    def get_target(self, l, targets, scaled_anchors, in_h, in_w):
        bs = len(targets)
        # 用于选取哪些先验框不包含物体
        # noobj_mask:[bs, 3, 13, 13]
        # 1：正样本；0：负样本
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        # 让网络更加去关注小目标
        # box_loss_scale:[bs, 3, 13, 13]
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        # y_true:[bs, 3, 13, 13, 25]
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad = False)
        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            # batch_target[3, 5]
            batch_target = torch.zeros_like(targets[b])
            
            # 计算出正样本在特征层上的中心点
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()

            # 真实框[num_true_box, 4]
            # 先验框[9, 4]
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2: 4]), 1))
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((len(scaled_anchors), 2)), torch.FloatTensor(scaled_anchors)), 1))
            
            # 计算IoU
            # best_ns：[每一个真实框最重合的先验框的序号]
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)
            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue
                # 判断这个先验框是当前特征点的哪一个先验框
                k = self.anchors_mask[l].index(best_n)
                # 获得真实框属于哪个网格点(左上角)
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                # 取出真实框的种类
                c = batch_target[t, 4].long()
                # noobj_mask代表无目标的特征点
                noobj_mask[b, k, j, i] = 0
                # tx、ty代表中心调整参数的真实值
                y_true[b, k, j, i, 0] = batch_target[t, 0]
                y_true[b, k, j, i, 1] = batch_target[t, 1]
                y_true[b, k, j, i, 2] = batch_target[t, 2]
                y_true[b, k, j, i, 3] = batch_target[t, 3]
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1
                # 大目标loss权重小，小目标loss权重大
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        """判断忽略哪些先验框"""
        bs = len(targets)

        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        # 生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # 计算调整后的先验框中心与宽高
        pred_boxes_x = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)

        for b in range(bs):           
            # pred_boxes_for_ignore:[num_anchors, 4]
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                #  计算出正样本在特征层上的中心点
                batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
                batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
                batch_target = batch_target[:, :4].type_as(x)
                #   计算交并比
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                #   每个先验框对应真实框的最大重合度
                anch_ious_max, _ = torch.max(anch_ious, dim = 0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes

    def forward(self, l, input, targets=None):
        # l:第几个特征层
        # input:[bs, 3*(5+num_classes), 13, 13]
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        stride_h = self.input_shape[0] / in_h # 416 // 13 = 32
        stride_w = self.input_shape[1] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors] # anchor缩放到特征层上
        
        # prediction:[bs, 3, 13, 13, 5+num_classes]
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        # 先验框参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 获得网络预测结果
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)
        y_true = y_true.type_as(x)
        noobj_mask = noobj_mask.type_as(x)
        box_loss_scale = box_loss_scale.type_as(x)

        box_loss_scale = 2 - box_loss_scale
        loss = 0
        obj_mask = y_true[..., 4] == 1 # 是否包含物体
        n = torch.sum(obj_mask)
        if n != 0:
            # giou: [bs, feat_w, feat_h, anchor_num, 1]
            giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
            loss_loc = torch.mean((1 - giou)[obj_mask])
            loss_cls = torch.mean(self.BCELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
        loss_conf = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss += loss_conf * self.balance[l] * self.obj_ratio
        return loss

def get_lr_scheduler(lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05):
    """学习率下降公式"""
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)))
        return lr

    warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
    warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
    no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
    func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """设置学习率"""
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr