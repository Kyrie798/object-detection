import numpy as np
from PIL import Image

def cvtColor(image):
    """将图像转换成RGB"""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert("RGB")
        return image 

def get_classes(classes_path):
    """获取类名和数量"""
    with open(classes_path, "r") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_anchors(anchors_path):
    """获取anchors和数量"""
    with open(anchors_path, "r") as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(",")]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

def preprocess_input(image):
    """归一化处理"""
    image /= 255.0
    return image 

def resize_image(image, size, letterbox_image):
    """对输入图像进行resize"""
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new("RGB", size, (128, 128, 128))
        new_image.paste(image, ((w-nw) // 2, (h-nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_lr(optimizer):
    """获取学习率"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]