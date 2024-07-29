import json
import cv2
import os
from hoi_label import hico_text_label, coco_class_dict, valid_obj_ids
import pickle

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

action_list_file = 'list_action.txt'
with open(action_list_file, 'r') as f:
     action_lines = f.readlines()

valid_verb = {}
for action_line in action_lines[2:]:
            act_id, act_name = action_line.split()
            act_id = int(act_id)
            valid_verb[act_id] = act_name

def convert_bbox(bbox, width, height) :
    new_bbox = [0,0,0,0]
    new_bbox[0] = bbox[0] / width
    new_bbox[1] = bbox[1] / height
    new_bbox[2] = bbox[2] / width
    new_bbox[3] = bbox[3] / height
    return new_bbox

with open("../../test_hico_sg3.json", "r") as file:
    json = json.load(file)

results = []
for file in json :
    out = {}
    out['file_name'] = file['file_name']
    out['img_id'] = int(file['file_name'].split('.jpg')[0].split('_')[2])
    if 'train' in file['file_name'].split('.jpg')[0].split('_')[1] :
        folder = 'train'
    else :
        folder = 'test'
    img = cv2.imread(os.path.join(f'/nas2/lait/1000_Members/pgb/hico/images/{folder}2015', file['file_name']))

    out['width'] = img.shape[1]
    out['height'] = img.shape[0]

    text = []
    prompt = ""
    sub_phr = []
    obj_phr = []
    act_phr = []

    sub_bbox_list = []
    obj_bbox_list = []
    for idx, hoi in enumerate(file['hoi_annotation']) :
        sub = hoi['subject_id']
        obj = hoi['object_id']
        verb_id = hoi['category_id']

        verb = hico_verb_dict_text[verb_id]
        if verb == 'and' : 
            verb = 'with'
        text_verb = valid_verb[verb_id]
        text_sub = coco_class_dict_text[file['annotations'][sub]['category_id']]
        text_obj = coco_class_dict_text[file['annotations'][obj]['category_id']]

        if idx == (len(file['hoi_annotation'])-1) : # final
            prompt += (f"a {text_sub} is {verb} a {text_obj}")
        else : 
            prompt += (f"a {text_sub} is {verb} a {text_obj}, ")
        text.append((text_verb,text_obj))
        sub_phr.append(text_sub)
        obj_phr.append(text_obj)
        act_phr.append(verb)

        sub_bbox = file['annotations'][sub]['bbox']
        obj_bbox = file['annotations'][obj]['bbox']
        #if file['file_name'] == 'HICO_test2015_00008020.jpg' : import pdb; pdb.set_trace()

        sub_bbox_out = convert_bbox(sub_bbox, out['width'], out['height'])
        obj_bbox_out = convert_bbox(obj_bbox, out['width'], out['height'])
        sub_bbox_list.append(sub_bbox_out)
        obj_bbox_list.append(obj_bbox_out)
    out['text'] = text
    out['prompt'] = prompt
    out['subject_phrases'] = sub_phr
    out['object_phrases'] = obj_phr
    out['action_phrases'] = act_phr
    out['subject_boxes'] = sub_bbox_list
    out['object_boxes'] = obj_bbox_list
    out['save_folder_name'] = 'hico-det-sg3'
    results.append(out)


output_file = 'hico_det_test_sg3_2.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(results, f)