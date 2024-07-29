import torch 
# from ldm.modules.encoders.modules import FrozenCLIPEmbedder
# from ldm.modules.encoders.modules import BERTEmbedder
from transformers import CLIPProcessor, CLIPModel
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import os 
import math
import clip 
from PIL import Image
from torchvision import transforms
import multiprocessing
from zipfile import ZipFile 
from io import BytesIO


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
            transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] )
        ])

    def fetch_zipfile(self, ziproot):
        pid = multiprocessing.current_process().pid # get pid of this process.
        if pid not in self.zip_dict:
            self.zip_dict[pid] = ZipFile(ziproot)
        zip_file = self.zip_dict[pid]
        return zip_file

    def fetch_image(self, file_name):
        if self.use_zip:
            zip_file = self.fetch_zipfile(self.image_root)
            image = Image.open( BytesIO(zip_file.read(file_name)) ).convert('RGB')
        else:
            image = Image.open(  os.path.join(self.image_root,file_name)   ).convert('RGB')
        return image


class GroundedTextImageDataset_Grounding(Base):
    def __init__(self, json_path, image_root):
        super().__init__(image_root)

        self.image_root = image_root
        
        with open(json_path, 'r') as f:
            json_raw = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
        self.data = json_raw["images"] # donot name it images, which is misleading
        self.annotations = json_raw["annotations"]

        self.data = {  datum["id"]:datum for datum in self.data }
    
    def __getitem__(self, idx):
        anno = self.annotations[idx]
        anno_id = anno["id"]
        X,Y,W,H = anno['bbox']

        caption = self.data[anno["image_id"]]["caption"]
        file_name = anno["file_name"]
        img = Image.open(self.image_root / img_anno['file_name']).convert('RGB')
        #image = self.fetch_image(file_name)
        image_crop = self.preprocess(image.crop((X,Y,X+W,Y+H)).resize((224,224), Image.BICUBIC))

        positive = ''
        for (start, end) in anno['tokens_positive']:
            positive += caption[start:end]
            positive += ' '       
        positive = positive[:-1]

        return {'positive':positive,  'anno_id':anno_id, 'image_crop':image_crop}
       
    def __len__(self):
        return len(self.annotations)


class HICO_DET(Base):
    def __init__(self, json_path, image_root):
        super().__init__(image_root)

        self.image_root = image_root
        with open(json_path, 'r') as f:
            self.annotations = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'

        # self.data = {datum["id"]:datum for datum in self.data }
    
    def __getitem__(self, idx):
        anno = self.annotations[idx]
        anno_id = idx+1
        X,Y,W,H = anno['bbox']

        # caption = self.data[anno["image_id"]]["caption"]
        file_name = anno["file_name"]
        img = Image.open(self.image_root / img_anno['file_name']).convert('RGB')
        image_crop = self.preprocess(image.crop((X,Y,X+W,Y+H)).resize((224,224), Image.BICUBIC))

        positive = ''
        import pdb; pdb.set_trace()
        # tokens_positive: starting and ending index for this entity in the caption. 
        for (start, end) in anno['tokens_positive']:
            positive += caption[start:end]
            positive += ' '       
        positive = positive[:-1]

        return {'positive':positive,  'anno_id':anno_id, 'image_crop':image_crop}
       
    def __len__(self):
        return len(self.annotations)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =#


# subject, object, action
@torch.no_grad()
def fire_clip_before_after(loader, folder):
    """
    This will save CLIP feature before/after projection. 

    before projection text feature is the one used by stable-diffsuion. 
    For before_projection, its feature is unmormalized. 
    For after_projection, which is CLIP aligned space, its feature is normalized.   

    You may want to use project / inv_project to project image feature into CLIP text space. (Haotian's idea)
    """
    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)

    os.makedirs(os.path.join(folder, 'text_features_before'),  exist_ok=True)
    os.makedirs(os.path.join(folder, 'text_features_after'),  exist_ok=True)
    os.makedirs(os.path.join(folder, 'image_features_before'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'image_features_after'), exist_ok=True)

    
    for batch in tqdm(loader):
        inputs = processor(text=batch['positive'],  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = batch['image_crop'].cuda() # we use our own preprocessing without center_crop 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)

        text_before_features = outputs.text_model_output.pooler_output # before projection feature
        text_after_features = outputs.text_embeds # normalized after projection feature (CLIP aligned space)

        image_before_features = outputs.vision_model_output.pooler_output # before projection feature
        image_after_features = outputs.image_embeds # normalized after projection feature (CLIP aligned space)

        for idx, text_before, text_after, image_before, image_after  in zip(batch["anno_id"], text_before_features, text_after_features, image_before_features, image_after_features):
            save_name = os.path.join(folder, 'text_features_before', str(int(idx)) )
            torch.save(text_before.clone().cpu(), save_name)

            save_name = os.path.join(folder, 'text_features_after', str(int(idx)) )
            torch.save(text_after.clone().cpu(), save_name)

            save_name = os.path.join(folder, 'image_features_before', str(int(idx)) )
            torch.save(image_before.clone().cpu(), save_name)

            save_name = os.path.join(folder, 'image_features_after', str(int(idx)) )
            torch.save(image_after.clone().cpu(), save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str,  default='/nas2/lait/1000_Members/pgb/hico/annotations/test_hico.json', help="")
    parser.add_argument("--image_root", type=str,  default='/nas2/lait/1000_Members/pgb/hico/images/test2015', help="")
    parser.add_argument("--folder", type=str,  default="out", help="")
    args = parser.parse_args()

    # json_path = '/nas2/lait/1000_Members/pgb/hico/annotations/test_hico.json'
    # image_root = '/nas2/lait/1000_Members/pgb/hico/images/test2015'

    # if args.total_chunk is not None:
    #     assert args.chunk_idx in list(range(args.total_chunk))

    dataset = HICO_DET(args.json_path, args.image_root)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=False)
    os.makedirs(args.folder, exist_ok=True)

    fire_clip_before_after(loader, args.folder)


