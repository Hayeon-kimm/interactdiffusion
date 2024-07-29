#!/usr/bin/env python
# coding: utf-8

# HERE
import torch
from transformers import CLIPProcessor, CLIPModel
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import clip
from PIL import Image
from torchvision import transforms
import multiprocessing
from zipfile import ZipFile
from io import BytesIO
import os
import math
from hoi_label import hico_text_label, coco_class_dict, valid_obj_ids
import sys
#from hico_det.hico_categories import HICO_ACTIONS, HICO_INTERACTIONS, NON_INTERACTION_IDS, HICO_OBJECTS, VERB_MAPPER


hico_verb_dict = {}
for k,v in hico_text_label.items():
    verb_idx, obj_idx = k[0], k[1]
    if v.split(' ')[6] == 'a' or v.split(' ')[6] == 'an':
        verb = v.split(' ')[5]
    else:
        verb = ' '.join(v.split(' ')[5:7])
    hico_verb_dict[verb_idx+1] = verb

hico_verb_dict_text = dict(zip(hico_verb_dict.keys(), hico_verb_dict.values()))
coco_class_dict_text = dict(zip(coco_class_dict.keys(), coco_class_dict.values()))

def get_categoty_name(id):
    return coco_class_dict_text[id]

def get_action_name(id):
    try:
        action = HICO_INTERACTIONS[id]['action']
    except:
        print(id)
    if action == "and":
        return "with"
    action_name = action.replace('_', ' ')
    new_action_name = []
    for str in action_name.split(' '):
        if str in VERB_MAPPER:
            str = VERB_MAPPER[str]
        new_action_name.append(str)
    return " ".join(new_action_name)

def xyxy_to_xywh(bbox):
    return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

def max_bbox(box1, box2):
    return [min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3])]


class Base():
    def __init__(self, image_root):
        self.image_root = image_root
        self.use_zip = True if image_root[-4:] == ".zip" else False
        if self.use_zip:
            self.zip_dict = {}

        # This is CLIP mean and std
        # Since our image is cropped from bounding box, thus we directly resize to 224*224 without center_crop to keep obj whole information. 
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def fetch_zipfile(self, ziproot):
        pid = multiprocessing.current_process().pid  # get pid of this process.
        if pid not in self.zip_dict:
            self.zip_dict[pid] = ZipFile(ziproot)
        zip_file = self.zip_dict[pid]
        return zip_file

    def fetch_image(self, file_name):
        if self.use_zip:
            zip_file = self.fetch_zipfile(self.image_root)
            image = Image.open(BytesIO(zip_file.read(file_name))).convert('RGB')
        else:
            image = Image.open(os.path.join(self.image_root, file_name)).convert('RGB')
        return image

class HICODetDataset_Detection(Base):
    def __init__(self, instances_json_path, image_root, chunk_idx, total_chunk):
        super().__init__(image_root)

        self.image_root = image_root
        self.instances_json_path = instances_json_path

        # Load all jsons
        with open(instances_json_path, 'r') as f:
            annos = json.load(f)
        
        version = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(version).cuda()
        self.processor = CLIPProcessor.from_pretrained(version)
        
        # clean_annotations(instances_data["annotations"])
        self.annotations = annos

        # Misc
        self.image_ids = []  # main list for selecting images
        self.image_id_to_filename = {}  # file names used to read image
        for anno in self.annotations:
            image_id = int(anno['file_name'].split('.jpg')[0].split('_')[-1])
            filename = anno['file_name']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename

    def __getitem__(self, index):
        anno = self.annotations[index]
        anno_id = int(anno['file_name'].split('.jpg')[0].split('_')[-1])
        filename = self.image_id_to_filename[anno_id]
        image = self.fetch_image(filename)

        classes = [obj["category_id"] for obj in anno["annotations"]]

        prompts = []
        hois = []

        for hoi in anno['hoi_annotation']:
            subject_annotation = anno['annotations'][hoi['subject_id']]
            subject_category_name = get_categoty_name(subject_annotation['category_id'])
            object_annotation = anno['annotations'][hoi['object_id']]
            object_category_name = get_categoty_name(object_annotation['category_id'])

            action_id = hoi["category_id"] - 1  # Starting from 1
            target_id = hoi["object_id"]

            sub = hoi['subject_id']
            obj = hoi['object_id'] # annotation idx
            verb_id = hoi['category_id']
            object_id = anno['annotations'][obj]['category_id'] # obj bbox annotations의 idx

            text_hoi = hico_text_label[(verb_id-1, valid_obj_ids.index(object_id))] # 'a photo of ~ '
            action_name = hico_verb_dict_text[verb_id]
            
            subject_image_crop = self.preprocess(image.crop(subject_annotation['bbox']).resize((224, 224), Image.BICUBIC))
            object_image_crop = self.preprocess(image.crop(object_annotation['bbox']).resize((224, 224), Image.BICUBIC))
            #max_bbox 확인해보기
            action_image_crop = self.preprocess(image.crop(max_bbox(subject_annotation['bbox'], object_annotation['bbox'])).resize((224, 224), Image.BICUBIC))  # not using
            prompts.append(f"a {subject_category_name} is {action_name} a {object_category_name}")

            
            with torch.no_grad():
                inputs = self.processor(text=[subject_category_name, object_category_name, action_name], return_tensors="pt", padding=True)
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['pixel_values'] = torch.stack([subject_image_crop, object_image_crop, action_image_crop]).cuda()  # we use our own preprocessing without center_crop
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                outputs = self.model(**inputs)

            text_before_features = outputs.text_model_output.pooler_output  # before projection feature
            text_after_features = outputs.text_embeds  # normalized after projection feature (CLIP aligned space)

            image_before_features = outputs.vision_model_output.pooler_output  # before projection feature
            image_after_features = outputs.image_embeds  # normalized after projection feature (CLIP aligned space)

            hois.append({
                'subject_xywh': xyxy_to_xywh(subject_annotation['bbox']),
                'object_xywh': xyxy_to_xywh(object_annotation['bbox']),
                'action': action_name,
                'subject': subject_category_name,
                'object': object_category_name,
                
                'subject_text_embedding_before': text_before_features[0].cpu(),
                'subject_text_embedding_after': text_after_features[0].cpu(),
                'subject_image_embedding_before': image_before_features[0].cpu(),  # not using
                'subject_image_embedding_after': image_after_features[0].cpu(),  # not using
                'object_text_embedding_before': text_before_features[1].cpu(),
                'object_text_embedding_after': text_after_features[1].cpu(),
                'object_image_embedding_before': image_before_features[1].cpu(),  # not using
                'object_image_embedding_after': image_after_features[1].cpu(),  # not using
                'action_text_embedding_before': text_before_features[2].cpu(),
                'action_text_embedding_after': text_after_features[2].cpu(),
                'action_image_embedding_before': image_before_features[2].cpu(),  # not using
                'action_image_embedding_after': image_after_features[2].cpu()  # not using
            })
            del image_before_features, image_after_features, text_before_features, text_after_features, outputs, inputs
        return {'file_name': anno['file_name'],
                'anno_id': anno_id,
                'image': image,
                'data_id': anno_id,
                'caption': ", ".join(prompts),
                'hois': hois
               }

    def __len__(self):
        return len(self.annotations)


dataset_root = "/nas2/lait/1000_Members/pgb/hico"
annotation_filename = "annotations/test_hico.json"
annotation_path = os.path.join(dataset_root, annotation_filename)
image_root = os.path.join(dataset_root, "images", "test2015")
dataset = HICODetDataset_Detection(annotation_path, image_root, None, None)

save_root = "hico_det_clip_test"
os.makedirs(save_root, exist_ok=True)
for d in tqdm(dataset):
    torch.save(d, os.path.join(save_root, f"embed_{d['anno_id']}.clip.pt"))