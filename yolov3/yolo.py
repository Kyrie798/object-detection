
import torch
import numpy as np
import torch.nn as nn
import colorsys

from PIL import ImageDraw, ImageFont

from model.yolo import YoloBody
from utils.utils import cvtColor, resize_image, preprocess_input, get_classes, get_anchors
from utils.utils_bbox import DecodeBox

class YOLO(object):
    def __init__(self):
        self.input_shape = [416, 416]

        self.model_path = "model_data/yolo_weights.pth"
        self.classes_path = "model_data/coco_classes.txt"
        self.anchors_path = "model_data/yolo_anchors.txt"
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = YoloBody(self.anchors_mask, self.num_classes)
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net = self.net.eval()
        self.net = nn.DataParallel(self.net)
        self.net = self.net.to(self.device)

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        # 给图像增加灰条，实现不失真的resize
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), False)
        # 添加batch维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).to(self.device)
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, False, conf_thres = 0.5, nms_thres = 0.3)
                                                    
            if results[0] is None: 
                return image

            top_label = np.array(results[0][:, 6], dtype = "int32")
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        # 设置字体与边框厚度
        font = ImageFont.truetype(font="model_data/simhei.ttf", size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32"))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        # 图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype("int32"))
            left = max(0, np.floor(left).astype("int32"))
            bottom = min(image.size[1], np.floor(bottom).astype("int32"))
            right = min(image.size[0], np.floor(right).astype("int32"))

            label = "{} {:.2f}".format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode("utf-8")
            # print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,"UTF-8"), fill=(0, 0, 0), font=font)
            del draw

        return image

            