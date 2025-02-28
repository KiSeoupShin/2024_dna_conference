# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import math
import logging
import functools
import braceexpand
import random
import pdb
import json

import pandas as pd
import numpy as np
import pyarrow as pa
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000                                                                                              

from typing import Union
from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
import torchvision.datasets as datasets
from torchvision.datasets.folder import DatasetFolder
import torchvision.datasets as datasets
import torchvision.transforms as T
from third_party.open_clip.clip import tokenize

from transformers import SamModel, SamProcessor, AutoProcessor, AutoModelForZeroShotObjectDetection
from copy import deepcopy
import torchvision

from io import BytesIO
import requests
import PIL

from imgnet_classes import IMAGENET2012_CLASSES

from eval_utils import SegmentImage

import sys
sys.path.append("/home/gisub/Desktop/2024_dna_conference/sam2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


## Structure of dataset directory
## CIRR: under ./data/CIRR
## validation images ./dev/
## caption split ./captions/cap.rc2.val.json
## image split ./image_splits/split.rc2.val.json
class CIRR(Dataset):
    def __init__(self, transforms, transforms_mask=None, is_mask=False,mode='caps', 
    vis_mode=False, test=False, root='./data'):
        self.mode = mode
        self.transforms = transforms
        self.transforms_mask = transforms_mask
        self.is_mask = is_mask
        self.vis_mode = vis_mode
        ## mode to use test split of CIRR
        self.test = test
        self.root = os.path.join(root, 'CIRR')
        self.root_img = os.path.join(self.root, 'dev')
        if self.test:
            self.root_img = os.path.join(self.root, 'test1')
            if self.mode == 'caps':
                self.json = os.path.join(self.root , 'captions/cap.rc2.test1.json')
            else:
                self.json = os.path.join(self.root, 'image_splits/split.rc2.test1.json')
        else:
            if self.mode == 'caps':
                self.json = os.path.join(self.root, 'captions/cap.rc2.val.json')
            else:
                self.json = os.path.join(self.root, 'image_splits/split.rc2.val.json')
        logging.debug(f'Loading json data from {self.json}.')
        data = json.load(open(self.json, "r"))                                
        self.ref_imgs = []
        self.target_imgs = []
        self.target_caps = []        
        if self.test:
            self.init_test(data)
        elif self.mode == 'caps':            
            self.init_val(data)                        
        else:
            self.target_imgs = [key + ".png" for key in data.keys()]                    
        if self.vis_mode:
            self.target_imgs = list(set(self.target_imgs))
        logging.info("Use {} imgs".format(len(self.target_imgs)))        

    def init_test(self, data):
        self.pairids = []
        if self.mode == 'caps':
            for d in data:
                ref_path = d['reference']+ ".png"
                self.ref_imgs.append(ref_path)
                self.target_caps.append(d['caption']) 
                self.pairids.append(d['pairid'])
                self.target_imgs.append('dummy')
        else:
            self.target_imgs = [key + ".png" for key in data.keys()]

    def init_val(self, data):
        for d in data:
            ref_path = d['reference']+ ".png"
            tar_path = d['target_hard']+ ".png"
            self.ref_imgs.append(ref_path)
            self.target_imgs.append(tar_path)
            self.target_caps.append(d['caption'])            
    
    def return_testdata(self, idx):
        if self.mode == 'caps':
                ref_path = str(self.ref_imgs[idx])
                img_path = os.path.join(self.root_img, ref_path)
                ref_images = self.transforms(Image.open(img_path))
                target_cap = self.target_caps[idx]
                text_with_blank_raw = 'a photo of * , {}'.format(target_cap)    
                caption_only = tokenize(target_cap)[0]
                text_with_blank = tokenize(text_with_blank_raw)[0]                 
                return ref_images, text_with_blank, \
                    caption_only, str(self.ref_imgs[idx]), \
                        self.pairids[idx], text_with_blank_raw
        else:
            tar_path = str(self.target_imgs[idx])
            img_path = Image.open(os.path.join(self.root_img, tar_path))
            target_images = self.transforms(img_path)
            return target_images, tar_path

    def return_valdata(self, idx):
        if self.mode == 'caps' and not self.vis_mode:
            ref_path = str(self.ref_imgs[idx])
            img_path = os.path.join(self.root_img, ref_path)
            ref_images = self.transforms(Image.open(img_path))
            target_cap = self.target_caps[idx]
            text_with_blank = 'a photo of * , {}'.format(target_cap)    
            caption_only = tokenize(target_cap)[0]
            ref_text_tokens = tokenize(text_with_blank)[0]                 
            if self.is_mask:
                alpha_path = img_path.replace('/dev/', '/mask/')
                ref_alphas = self.transforms_mask(Image.open(alpha_path))
                return ref_images, ref_alphas[:1, :, :], ref_text_tokens, caption_only, \
                str(self.ref_imgs[idx]), str(self.target_imgs[idx]), \
                    target_cap  
            else:
                return ref_images, ref_text_tokens, caption_only, \
                    str(self.ref_imgs[idx]), str(self.target_imgs[idx]), \
                        target_cap                       
        else:
            tar_path = str(self.target_imgs[idx])
            img_path = os.path.join(self.root_img, tar_path)
            target_images = self.transforms(Image.open(img_path))
            return target_images, img_path

    def __getitem__(self, idx):
        if self.test:                        
            return self.return_testdata(idx)
        else:
            return self.return_valdata(idx)
    
    def __len__(self):
        return len(self.target_imgs)
        
## Fashion-IQ: under ./data/fashion-iq
## validation images ./images
## caption split ./json/cap.{cloth_type}.val.json, cloth_type in [toptee, shirt, dress]
## image split ./image_splits/split.{cloth_type}.val.json, cloth_type in [toptee, shirt, dress]
class FashionIQ(Dataset):
    def __init__(self, cloth, transforms, is_train=False, vis_mode=False, \
        mode='caps', is_return_target_path=False, root='./data'):
        root_iq = os.path.join(root, 'fashion-iq')
        self.root_img = os.path.join(root_iq, 'images')
        self.vis_mode = vis_mode
        self.mode = mode
        self.is_return_target_path = is_return_target_path
        self.transforms = transforms
        self.segment_model = SegmentImage()
        self.cloth = cloth
        if mode == 'imgs':
            self.json_file = os.path.join(root_iq, 'image_splits', \
                'split.{}.val.json'.format(cloth))
        else:
            self.json_file = os.path.join(root_iq, 'json', \
                'cap.{}.val.json'.format(cloth))                
        logging.debug(f'Loading json data from {self.json_file}.')

        self.ref_imgs = []
        self.target_imgs = []
        self.ref_caps = []
        self.target_caps = []        
        if mode == 'imgs':
            self.init_imgs()
            logging.info("Use {} imgs".format(len(self.target_imgs)))
        else:
            self.init_data()     
            logging.info("Use {} imgs".format(len(self.target_imgs)))

    def init_imgs(self):
        data = json.load(open(self.json_file, "r"))
        self.target_imgs = [key + ".jpg" for key in data]        

    def init_data(self):
        def load_data(data):
            for d in data:
                ref_path = os.path.join(self.root_img, d['candidate']+ ".jpg") 
                tar_path = os.path.join(self.root_img, d['target']+ ".jpg")            
                try:
                    Image.open(ref_path)
                    Image.open(tar_path)
                    self.ref_imgs.append(ref_path)
                    self.target_imgs.append(tar_path)
                    self.ref_caps.append((d['captions'][0], d['captions'][1]))
                    #self.target_caps.append(d['captions'][1])
                except:                
                    print('cannot load {}'.format(d['candidate']))
        if isinstance(self.json_file, str):
            data = json.load(open(self.json_file, "r"))        
            load_data(data)            
        elif isinstance(self.json_file, list):
            for filename in self.json_file:
                data = json.load(open(filename, "r")) 
                load_data(data)         

    def __len__(self):
        if self.mode == 'caps':
            return len(self.ref_imgs)
        else:
            return len(self.target_imgs)

    def return_imgs(self, idx):
        tar_path = str(self.target_imgs[idx])
        img_path = os.path.join(self.root_img, tar_path)
        target_images = self.transforms(Image.open(img_path))
        return target_images, os.path.join(self.root_img, tar_path)

    def return_all(self, idx):
        if self.vis_mode:
            tar_path = str(self.target_imgs[idx])
            target_images = self.transforms(Image.open(tar_path))
            return target_images, tar_path            
        ref_images = self.transforms(Image.open(str(self.ref_imgs[idx])))
        ref_alphas = self.segment_model.transform(str(self.ref_imgs[idx]), f"A {self.cloth}.")
        target_images = self.transforms(Image.open(str(self.target_imgs[idx])))
        cap1, cap2 = self.ref_caps[idx]
        text_with_blank = 'a photo of * , {} and {}'.format(cap2, cap1)
        token_texts = tokenize(text_with_blank)[0]                
        if self.is_return_target_path:
            return ref_images, ref_alphas, f"a photo of {self.cloth} that {cap1}", target_images, token_texts, token_texts, \
                str(self.target_imgs[idx]), str(self.ref_imgs[idx]), \
                    cap1
        else:
            return ref_images, target_images, text_with_blank


    def __getitem__(self, idx):
        if self.mode == 'imgs':            
            return self.return_imgs(idx)
        else:            
            return self.return_all(idx)
        
## COCO: under ./data/coco
## validation images ./val2017
## validation masked images ./val2017_masked
## validation csv file ./coco_eval.csv
class CsvCOCO(Dataset):
    def __init__(self, transforms, transforms_region, sep=",",
                return_data_identifier=False, return_filename=False, 
                root='./data'):
        self.transforms = transforms
        self.transforms_region = transforms_region
        self.root = os.path.join(root, 'coco')
        self.root_img = os.path.join(self.root, 'val2017')
        self.csv_file = os.path.join(self.root, 'coco_eval.csv')
        logging.debug(f'Loading csv data from {self.csv_file}.')
        df = pd.read_csv(self.csv_file, sep=sep)                
        self.images = df['id'].tolist()
        ## query_region contains the box of query regions.
        regions = df['query_regions'].tolist()
        self.regions = []
        for region in regions:
            x1, y1, x2, y2 = map(lambda x: int(float(x)), region.split(";"))
            self.regions.append([x1, y1, x2, y2])

        ## query_classes contains the class of query region in the target.
        self.query_classes = df['query_class'].tolist()
        self.classes = []
        ## classes contains the list of classes in the target.
        for list_class in df['classes'].tolist():
            if isinstance(list_class, str):
                list_class = list_class.split(";")
                self.classes.append(list_class)
            else:
                self.classes.append([""])        
        self.return_data_identifier = return_data_identifier
        logging.debug('Done loading data.')
        self.return_filename = return_filename

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_img, str(self.images[idx]))
        image = Image.open(img_path)        
        masked_path = os.path.join(self.root_img.replace('val2017', 'val2017_masked'), \
            str(self.images[idx]))
        image_masked = Image.open(masked_path)
        
        ## extract query region.
        x1, y1, x2, y2 = self.regions[idx]        
        region_image = image_masked.crop((x1, y1, x2, y2))

        image = self.transforms(image)
        ## no cropping is applied to query region.
        region_image = self.transforms_region(region_image)
        mask_tensor = (region_image > 0).to(torch.uint8)[0].unsqueeze(0).to(torch.float)
        query_class = self.query_classes[idx]
        other_classes = self.classes[idx]        
        text_with_blank = 'a photo of * and {}'.format(" and ".join(other_classes))
        text_with_queryclass = 'a photo of * and {} and {}'.format(query_class, \
            " and ".join(other_classes))
        raw_text = text_with_queryclass
        text_full = 'a photo of {} and {}'.format(query_class, " and ".join(other_classes))        
        text_with_blank = tokenize(text_with_blank)[0]
        text_with_queryclass = tokenize(text_with_queryclass)[0]
        text_full = tokenize(text_full)[0]
        return image, region_image, mask_tensor, text_full, text_with_blank, \
            text_with_queryclass, str(self.images[idx]), raw_text


class ImageList(Dataset):
    def __init__(self, input_filename, transforms, transforms_mask=None, root=None, 
                 return_filename=False, is_labels=False, is_mask=False):
        logging.debug(f'Loading txt data from {input_filename}.')
        with open(input_filename, 'r') as f:
            lines = f.readlines()
        if not is_labels:
            self.images = [line.strip() for line in lines]
        elif not is_mask:
            filenames = [line.strip() for line in lines]
            self.images = [name.split(" ")[0] for name in filenames] 
            self.labels = [int(name.split(" ")[1]) for name in filenames]
        else:
            filenames = [line.strip() for line in lines]
            self.images = [name.split(" ")[0] for name in filenames] 
            self.alphas = [name.split(" ")[1] for name in filenames] 
            self.labels = [int(name.split(" ")[2]) for name in filenames]
            self.transforms_mask = transforms_mask
        
        self.is_labels = is_labels
        self.is_mask = is_mask
        self.transforms = transforms
        self.root = root
        logging.debug('Done loading data.')
        self.return_filename = return_filename

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.root is not None:
            img_path = os.path.join(self.root, str(self.images[idx]))
        else:
            img_path = str(self.images[idx])
        
        if self.is_mask:
            alpha_path = os.path.join(self.root, str(self.alphas[idx]))
            alphas = self.transforms_mask(Image.open(alpha_path))
            alphas = alphas[0, :, :].unsqueeze(0)

        images = self.transforms(Image.open(img_path))

        class_name = img_path.split("/")[-2]
        noun = 'a photo of ' + IMAGENET2012_CLASSES[class_name]
        if self.return_filename:
            return images, img_path
        elif self.is_mask:
            target = self.labels[idx]
            return images, alphas, target, noun
        elif self.is_labels:
            target = self.labels[idx]
            return images, target       
        else:
            return images


class CustomFolder(Dataset):
    def __init__(self, folder, transform):
        image_lists = os.listdir(folder)
        self.samples = [os.path.join(folder, name) for name in image_lists]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = Image.open(str(path))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t",
                 return_data_identifier=False, return_filename=False):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.return_data_identifier = return_data_identifier
        logging.debug('Done loading data of {} samples'.format(len(self.images)))
        self.return_filename = return_filename

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        if self.return_filename:
            return images, str(self.images[idx])
        texts = tokenize([str(self.captions[idx])])[0]

        if self.return_data_identifier:
            return images, texts, 0
        return images, texts
    

class GRITDataset(Dataset):
    def __init__(self, input_filename, transforms):
        self.dataset = pd.read_csv(input_filename, sep='|')
        self.transforms = transforms

        self.device = 'cuda'
        # self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        # self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        self.checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam_model = build_sam2(self.model_cfg, self.checkpoint)
        self.sam_model.to(self.device)
        self.sam_processor = SAM2ImagePredictor(self.sam_model)

        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(0.5, 0.26)
        ])
    

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]

        image = Image.open(data['url']).convert("RGB")
        if image.width < 224 or image.height < 224:
            return None
        
        caption = data['caption']
        boxes = [[[float(i) for i in data['boxes'][1:-1].split(',')]]]
        mask = self.get_masks(image, boxes)

        image = self.transforms(image).to(self.device)

        mask = self.transforms.transforms[0](mask)
        mask = self.transforms.transforms[1](mask)
        mask = np.array(mask)

        binary_maskes = (mask[0, :, :] != 0)
        alpha = self.mask_transform((binary_maskes * 255).astype(np.uint8))

        return image, caption, alpha, data['url'], data['noun']
    

    def get_masks(self, image, boxes):
        with torch.inference_mode():
            self.sam_processor.set_image(image)
            masks, scores, _ = self.sam_processor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
                )
        
        return torch.tensor(masks)


    # def get_masks(self, image, boxes, sam_processor, sam_model, device):
    #     inputs = sam_processor(image, input_boxes=boxes, return_tensors="pt").to(device)
    #     with torch.no_grad():
    #         outputs = sam_model(**inputs)

    #     masks = sam_processor.image_processor.post_process_masks(
    #         outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    #     )

    #     combined_mask = None
    #     for mask in masks[0]:
    #         new_mask = mask.float()
    #         if combined_mask is None:
    #             combined_mask = new_mask
    #         else:
    #             combined_mask = np.maximum(combined_mask, new_mask)

    #     return combined_mask


class FeatDataset(Dataset):
    def __init__(self, input_filename):
        self.dataset = pd.read_csv(input_filename, sep='|')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        try:
            image_features = torch.tensor(np.load(data['url']))
        except:
            return None

        return image_features, data['noun'], data['caption']


class AlphaDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t",
                 return_data_identifier=False, return_filename=False):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.return_data_identifier = return_data_identifier
        logging.debug('Done loading data of {} samples'.format(len(self.images)))
        self.return_filename = return_filename

        self.device = "cuda:0"
        self.dino_model, self.dino_processor = self.get_dino()
        self.sam_model, self.sam_processor = self.get_sam()

        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(0.5, 0.26)
        ])


    def __len__(self):
        return len(self.captions)


    def __getitem__(self, idx):
        images = Image.open(str(self.images[idx]))
        texts = str(self.captions[idx])

        input_boxes = self.dino_process(images, texts)
        image_maskes = self.sam_process(images, input_boxes)

        images = self.transforms(images).to(self.device)

        image_maskes = self.transforms.transforms[0](image_maskes)
        image_maskes = self.transforms.transforms[1](image_maskes)
        image_maskes = np.array(image_maskes)

        binary_maskes = (image_maskes[:, :, 0] == 255)
        alphas = self.mask_transform((binary_maskes * 255).astype(np.uint8))

        return images, texts, alphas
    

    def get_dino(self):
        model_id = "IDEA-Research/grounding-dino-tiny"

        dino_processor = AutoProcessor.from_pretrained(model_id)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

        return dino_model, dino_processor
    

    def dino_process(self, images, texts):
        inputs = self.dino_processor(images=images, text=texts, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[images.size[::-1]]
        )

        return results[0]['boxes'].unsqueeze(0).cpu().numpy().tolist()
    

    def get_sam(self):
        sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

        return sam_model, sam_processor
    

    def sam_process(self, images, input_boxes):
        inputs = self.sam_processor(images, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )

        combined_mask = None
        for mask in masks[0]:
            new_mask = mask.float()
            if combined_mask is None:
                combined_mask = new_mask
            else:
                combined_mask = np.maximum(combined_mask, new_mask)
        
        return combined_mask


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def preprocess_txt(text):
    return tokenize([str(text)])[0]

def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    sizes = json.load(open(sizes_filename, 'r'))
    total_size = sum(
        [int(sizes[os.path.basename(shard)]) for shard in shards_list])
    num_shards = len(shards_list)
    return total_size, num_shards

def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path  = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )
    return DataInfo(dataloader, sampler)

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches

def get_csv_dataset(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
        
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_alpha_dataset(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = AlphaDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
        
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_feat_dataset(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = FeatDataset(input_filename=input_filename)
        
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        # pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=collate_fn_skip_none,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def collate_fn_skip_none(batch):
    # None 값을 필터링하여 배치 생성
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # 모든 아이템이 None인 경우
    return default_collate(batch)


def get_grit_dataset(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = GRITDataset(
        input_filename=input_filename,
        transforms=preprocess_fn)
        
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        # pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=collate_fn_skip_none,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_imgnet_r(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    path_data = os.path.join(args.root_data, 'imgnet/imagenet-r')
    dataset = CustomFolder(path_data, transform=preprocess_fn)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)


def get_directory_dataset(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CustomFolder(
        input_filename,
         transform=preprocess_fn)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == 'imgnet_r':
        return get_imgnet_r
    elif dataset_type == 'fashion-iq':
        return get_fashion_iq
    elif dataset_type == 'cirr':
        return get_cirr
    elif dataset_type == 'directory':
        return get_directory_dataset
    elif dataset_type == "csv":
        return get_csv_dataset  
    elif dataset_type == 'alpha':
        return get_alpha_dataset 
    elif dataset_type == 'grit':
        return get_grit_dataset
    elif dataset_type == 'feat':
        return get_feat_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    dataset_type_val = getattr(args, 'dataset_type_val', args.dataset_type)
    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args, preprocess_train, is_train=True)
    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, dataset_type_val)(
            args, preprocess_val, is_train=False)
    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")
    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")
    return data
