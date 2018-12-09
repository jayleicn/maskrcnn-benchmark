import os

import json
import torch
import torch.utils.data
from PIL import Image


from maskrcnn_benchmark.structures.bounding_box import BoxList


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_id2label(file_path):
    raw_id2label = load_json(file_path)
    return {k: v[0] for k, v in raw_id2label.items()}


class TVQACocoDetDataset(torch.utils.data.Dataset):
    def __init__(self, anno_file_format, split, img_root, transforms=None):
        self.img_root = img_root
        self.split = split  # train, valid, test
        self.transforms = transforms

        self.annotations = load_json(anno_file_format.format(split))

    def __getitem__(self, index):
        img_anno = self.annotations[index]
        img = Image.open(os.path.join(self._img_dir, img_anno["img_path"])).convert("RGB")
        target = self.get_target(index)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.annotations)

    def get_target(self, index):
        anno = self.annotations[index]
        width, height = anno["img_size"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", [int(e) for e in anno["label_ids"]])
        return target

    def get_img_info(self, index):
        img_anno = self.annotations[index]
        width, height = img_anno["img_size"]
        return {"width": width, "height": height}

    def map_class_id_to_class_name(self, class_id):
        # 0 should always be the background class
        return self.id2label[class_id]
