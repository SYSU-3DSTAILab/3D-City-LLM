import glob
import json
import os
import random
import re

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST, 
                    SHORT_QUESTION_LIST, DENSE_CAPTION_LIST, DENSE_CAPTION_ANSWER_LIST)

from lib.config import CONF
from data.sensaturban.model_util_sensaturban import SensatUrbanDatasetConfig

from PIL import Image
import rasterio
from rasterio.plot import show
from utils.pc_utils import random_sampling, rotx, roty, rotz

import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch= 500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.dense_caption_list = DENSE_CAPTION_LIST
        self.dense_caption_answer_list = DENSE_CAPTION_ANSWER_LIST
        self.answer_list = ANSWER_LIST

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        jsons = []
        if reason_seg_data == 'ReasonSeg':
            for split in splits:
                images_split = glob.glob(
                    os.path.join(
                        base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                    )
                )
                images.extend(images_split)
            jsons = [path.replace(".jpg", ".json") for path in images]
        elif reason_seg_data == 'rsvg':
            for split in splits:
                jsons = glob.glob(
                    os.path.join(
                        base_image_dir, 'RSVG', reason_seg_data, 'Json', split, '*.json'
                    )
                )
            imagePath = os.path.join(
                base_image_dir,'RSVG', reason_seg_data, 'images'
            )
            for path in jsons:
                name = os.path.basename(path)
                name = name.replace('.json', '.jpg')
                if '(' in name:
                    name = re.sub('\([0-9]\)', '', name)
                images.append(os.path.join(imagePath, name))
        elif reason_seg_data == 'SensatUrban-LISA-EX':
          for split in splits:
            images_split = glob.glob(
              os.path.join(
                base_image_dir,'SensatUrban-LISA-EX',split,'*.jpg'
              )
            ) 
            images.extend(images_split)
          jsons = [path.replace(".jpg", ".json") for path in images]
        
        elif reason_seg_data == 'UrbanBIS':
          for split in splits:
            json_split = glob.glob(
              os.path.join(
                base_image_dir,'UrbanBIS',split,'*.json'
              )
            )
            jsons.extend(json_split)
          images = [ 
            x.split('_')[0] + "_" + x.split('_')[1]+'.tif' 
            for x in jsons
          ]
          print(images[0])
          

        self.reason_seg_data = (images, jsons)
        print("number of reason_seg samples: ", len(images))

        if explanatory != -1 and reason_seg_data == 'ReasonSeg':
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            with open(
                os.path.join(
                    base_image_dir,
                    "reason_seg",
                    reason_seg_data,
                    "explanatory",
                    "train.json",
                )
            ) as f:
                items = json.load(f)
            for item in items:
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }

            print("len(self.img_to_explanation): ", len(self.img_to_explanation))

        print("preparing data...")
        DC = SensatUrbanDatasetConfig()
        
        base_meta_dir = "./data/meta_data/Train" # TODO: change this

        all_json_paths = glob.glob(os.path.join(base_meta_dir, "*", "*.json"))

        num_json_to_load = 1
        selected_jsons = random.sample(all_json_paths, num_json_to_load)

        SCANREFER_TRAIN = []
        for json_path in selected_jsons:
          try:
            data = json.load(open(json_path))
            if isinstance(data, list):
              SCANREFER_TRAIN.extend(data)
            elif isinstance(data, dict) and "data" in data:
              SCANREFER_TRAIN.extend(data["data"])
            else:
              SCANREFER_TRAIN.append(data)
          except Exception as e:
            print(f"Error")

        self.base_meta_dir = "./data/meta_data/Train" # TODO: change this

        self.all_json_paths = glob.glob(os.path.join(self.base_meta_dir, "**", "*.json"), recursive=True)
        if len(self.all_json_paths) == 0:
            raise FileNotFoundError(f"No JSON files found in {self.base_meta_dir}")
        else:
            print(f"Found {len(self.all_json_paths)} JSON files in total")

        self.json_cache = {}

        SCANREFER_VAL = json.load(open("./data/meta_data/Val/Object_Caption/SensatUrban_Object_Caption_Part_1_Val.json")) # TODO: change this

        self.alt_meta_dir = "./data/meta_data/Train/Object_Caption" # TODO: change this

        self.all_json_paths_B = glob.glob(
            os.path.join(self.alt_meta_dir, "*.json")
        )

        if len(self.all_json_paths_B) == 0:
            raise FileNotFoundError(f"No JSON files found in {self.alt_meta_dir}")
        else:
            print(f"Found {len(self.all_json_paths_B)} JSON files in B")

        scanrefer_train, scanrefer_val, all_scene_list= get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, -1)
        scanrefer = {"train": scanrefer_train, "val": scanrefer_val}
        
        self.scanrefer=scanrefer["train"]
        self.DC=DC
        self.other_object_cat = -1
        # self._load_sensaturban_data()     
        self.CONF = CONF
        self.use_cache = False
        self.cache = {}
        self.grounding_num_cands = 3
        self.scanrefer_all_scene = all_scene_list
        self.num_points = -1
        self.use_color = True
        self.use_height = True
        self.augment = False
        self.use_landmark = True
        self.num_inst_points = 4096
        self.num_scene_points = 4096
        self.other_object_cat = -1
        self.voxel_size_ap = 0.33
        self.voxel_size_glp = 0.33
        self.no_gt_instance = 192
        self.max_num_object = 600
        self.max_num_landmark = 192
        self.split = "train"
        self.no_gt_instance = False
        self.K = 5
        self.landmark_K = 10
        self.Fine_grained_building_category = {'Commercial': 0, 'Residential': 1, 'Office': 2, 'Cultural': 3, 'Transportation': 4, 'Municipal': 5, 'Temporary': 6, 'Unclassified': 7}

    def _get_raw2label(self, SCANNET_V2_TSV):
        # Mapping
        scannet_labels = self.DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = int(data["object_id"])
            object_name = " ".join(data["object_name"].split("_"))

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except (KeyError, TypeError):
                    all_sem_labels[scene_id].append(self.other_object_cat)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = int(data["object_id"])
            object_name = data["object_name"]
            ann_id = int(data["ann_id"])

            try:
                sem_label = self.raw2label[object_name]
            except (KeyError, TypeError):
                sem_label = self.other_object_cat

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _load_sensaturban_data(self):
        print("Loading data...")
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))
        self.raw2label = {self.DC.type2class[class_ind]:class_ind for class_ind in self.DC.type2class.keys()}
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]
        # translation factors
        factor = (torch.rand(3) - 0.5).tolist()
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox


    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ### For original llava 
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)

        ### Getting item for City-LLM-1.2M dataset  
        if random.random() < 0.75:
            selected_json_path = random.choice(self.all_json_paths)
        else:
            selected_json_path = random.choice(self.all_json_paths_B)


        if selected_json_path in self.json_cache:
            items = self.json_cache[selected_json_path]
        else:
            with open(selected_json_path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict) and "data" in data:
                items = data["data"]
            elif isinstance(data, list):
                items = data
            else:
                items = [data]

            self.json_cache[selected_json_path] = items

        sample = random.choice(items)
        idx_2 = random.randint(0, len(self.scanrefer) - 1)

        scene_inf = sample["scene_id"]
        obj_id = sample["object_id"]
        task_type = sample["type"]
        question = sample["question"]
        answer = sample["answer"]
        
        MAX_NUM_OBJ = self.max_num_object
        MAX_NUM_LANDMARK = self.max_num_landmark
        if obj_id != "-" and obj_id.isdigit():
          object_id = int(obj_id)
        else:
          object_id = None

        object_prompt = ""
        landmark_prompt = ""
        semantic_prompt = ""

        scene_parts = scene_inf.split("-", 1)
        if len(scene_parts) == 2:
            scene_name, scene_id = scene_parts

        local_image = np.zeros((3, 9, 9))
        global_image = np.zeros((3, 9, 9))
        lang_gt = 1
        pc_feats = np.zeros((1, 1024))
        landmark_feats = np.zeros((1, 768))
        pc_feats_nearest = np.zeros((5, 1024))
        landmark_feats_nearest = np.zeros((5, 768))
        
        ### Multi-modality Embedding for SensatUrban Dataset
        if scene_name == "SensatUrban":

          if self.use_cache and scene_id in self.cache:
            mesh_vertices, instance_labels, semantic_labels, instance_bboxes, landmark_names, landmark_ids, globalShift = self.cache[scene_id]
          else:
            pg_file = os.path.join(self.CONF.PATH.SensatUrban_PG, scene_id + ".pth")
            feat_file = os.path.join(self.CONF.PATH.SensatUrban_FEAT, scene_id+".json")
            landmark_file = os.path.join(self.CONF.PATH.SensatUrban_LANDMARK, scene_id+".landmark_feat.json")
            landmark_nearest_file = os.path.join(self.CONF.PATH.SensatUrban_LANDMARK, scene_id+".landmark_feat.json")
            landmark_name_file = os.path.join(self.CONF.PATH.SensatUrban_BOX, scene_id+"_bbox.json")

            coords, colors, label_ids, instance_ids, label_ids_pg, instance_ids_pg, instance_bboxes, \
                landmark_names, landmark_ids, globalShift = torch.load(pg_file, weights_only = False)
            
            with open(feat_file, 'r') as json_file:
                feats = json.load(json_file)
            
            with open(landmark_file, 'r') as json_file:
                landmark_feat = json.load(json_file)

            with open(landmark_nearest_file, 'r') as json_file:
                landmark_nearest_feat = json.load(json_file)

            with open(landmark_name_file, 'r') as json_file:
                landmark_name_idx_file = json.load(json_file)

            mesh_vertices = np.concatenate([coords, colors], axis=1) 
            instance_bboxes = np.stack([instance_bboxes[instance_id] for instance_id in sorted(instance_bboxes.keys()) if instance_id != -100])
            if self.no_gt_instance:
                instance_labels = instance_ids_pg
                semantic_labels = label_ids_pg
            else:                
                instance_labels = instance_ids
                semantic_labels = label_ids
            if self.use_cache:
                self.cache[scene_id] = mesh_vertices, instance_labels, semantic_labels, instance_bboxes, \
                    landmark_names, landmark_ids, globalShift

          if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]
          else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:6] = point_cloud[:, 3:6] / 127.5 - 1 

          if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)
            
          if self.num_points > 0:
            point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
            instance_labels = instance_labels[choices]
            semantic_labels = semantic_labels[choices]

          instance_class = []
          pc_feats = []

          cand_instance_ids = [cand_id for cand_id in np.unique(instance_labels) if cand_id != -100]
          nearest_instance_ids = find_nearest_instances(instance_bboxes, object_id, self.K)
          nearest_instance_landmark_ids = find_nearest_instances(instance_bboxes, object_id, self.landmark_K)
        
          ## For Object-level and Relationship-level Tasks
          item_id = object_id

          # Image - Top view feat
          raster_file = os.path.join(self.CONF.PATH.SensatUrban_MAP, scene_id+".tif")
          image = safe_read_raster(raster_file)
          instance_bboxes_pixel = np.copy(instance_bboxes)
          instance_bboxes_pixel[:,0] = instance_bboxes_pixel[:,0] / 0.1
          instance_bboxes_pixel[:,1] = instance_bboxes_pixel[:,1] / 0.1
          instance_bboxes_pixel[:,3] = instance_bboxes_pixel[:,3] / 0.1
          instance_bboxes_pixel[:,4] = instance_bboxes_pixel[:,4] / 0.1
          arr = instance_bboxes[:,-1]
          row = find_rows_with_value(arr, item_id)

          if len(row) == 0:
            subimage = np.zeros((256,256,3))
          else:         
            X_pixel = int(instance_bboxes_pixel[row, 0])
            Y_pixel = int(instance_bboxes_pixel[row, 1])
            X_delta = int(instance_bboxes_pixel[row, 3])
            Y_delta = int(instance_bboxes_pixel[row, 4])

            bbox = instance_bboxes[row, :]
            bbox = np.array(bbox).reshape(-1)
            x, y, z = bbox[0], bbox[1], bbox[2]
            dx, dy, dz = bbox[3], bbox[4], bbox[5]

            object_prompt = (
              f"The object is located at coordinates "
              f"({x:.2f}, {y:.2f}, {z:.2f}) meters, "
              f"with spatial spans of {dx:.2f} meters along X, "
              f"{dy:.2f} meters along Y, and {dz:.2f} meters along Z."
            )

            semantic_id = int(bbox[-2])

            if semantic_id == 2:
                semantic_name = "Building"
            elif semantic_id == 9:
                semantic_name = "Car"
            elif semantic_id == 5:
                semantic_name = "Parking"
            elif semantic_id == 0:
                semantic_name = "Ground"
            else:
                semantic_name = "Unknown"
            semantic_prompt = f" This object belongs to {semantic_name}."
            
            Image_LH = 3 * max(X_delta, Y_delta)

            image_with_rectangle = image[:, :, :]
            x1 = int(image.shape[2] - Y_pixel - int(Y_delta/2))
            y1 = int(X_pixel - int(X_delta/2))
            x2 = int(image.shape[2] - Y_pixel + int(Y_delta/2))
            y2 = int(X_pixel + int(X_delta/2))
            thickness = min(15, x2 - x1, y2 - y1)

            if min(x2 - x1, y2 - y1) < 40:
              Image_LH = 10 * max(X_delta, Y_delta)
              thickness = 4
            
            for t in range(thickness):
              try:
                image_with_rectangle = image[:, :, :]
                image_with_rectangle[0, x1:x2, y1 + t + 1] = 255
                image_with_rectangle[0, x1:x2, y2 - t - 1] = 255
                image_with_rectangle[0, x1 + t + 1, y1:y2] = 255
                image_with_rectangle[0, x2 - t - 1, y1:y2] = 255
                image_with_rectangle[1, x1:x2, y1 + t + 1] = 0
                image_with_rectangle[1, x1:x2, y2 - t - 1] = 0
                image_with_rectangle[1, x1 + t + 1, y1:y2] = 0
                image_with_rectangle[1, x2 - t - 1, y1:y2] = 0
                image_with_rectangle[2, x1:x2, y1 + t + 1] = 0
                image_with_rectangle[2, x1:x2, y2 - t - 1] = 0
                image_with_rectangle[2, x1 + t + 1, y1:y2] = 0
                image_with_rectangle[2, x2 - t - 1, y1:y2] = 0
              except IndexError:
                continue
            
            local_image = extract_subimage(image_with_rectangle, image_with_rectangle.shape[2] - Y_pixel, X_pixel, Image_LH, Image_LH)
        
          local_image = np.transpose(local_image, (1, 2, 0))
          global_image = np.transpose(image, (1, 2, 0))
        
          # PC - Uni3d feat
          pc_feats = []
          landmark_feats = []
          
          try:
            feat_instance = feats[str(int(item_id))]
          except (KeyError, TypeError):
            feat_instance = np.zeros((1, 1024))
          pc_feats.append(feat_instance)
          pc_feats = np.vstack(pc_feats)

          try:
            landmark_feat_instance = landmark_feat[str(int(item_id))]
          except (KeyError, TypeError):
            landmark_feat_instance = np.zeros((1, 768))
          landmark_feats.append(landmark_feat_instance)
          landmark_feats = np.vstack(landmark_feats)

          while len(nearest_instance_ids) < self.K:
            nearest_instance_ids.append(object_id)

          if isinstance(nearest_instance_landmark_ids, np.ndarray):
            nearest_instance_landmark_ids = nearest_instance_landmark_ids.tolist()

          while len(nearest_instance_landmark_ids) < self.landmark_K: 
            nearest_instance_landmark_ids.append(object_id)

          pc_feats_nearest = []
          for i in nearest_instance_ids:
            feat_instance = np.zeros((1, 1024))
            try:
                feat_instance = feats[str(int(i))]
            except (KeyError, TypeError):
                feat_instance = np.zeros((1, 1024))
            pc_feats_nearest.append(feat_instance)
          pc_feats_nearest = np.vstack(pc_feats_nearest)

          landmark_feats_nearest = []
          for i in nearest_instance_ids:
            landmark_feat_instance = np.zeros((1, 768))
            try:
                landmark_feat_instance = landmark_nearest_feat[str(int(i))]
            except (KeyError, TypeError):
                landmark_feat_instance = np.zeros((1, 768))
            landmark_feats_nearest.append(landmark_feat_instance)
          landmark_feats_nearest = np.vstack(landmark_feats_nearest)


          landmark_names_nearest = []
          for i in nearest_instance_landmark_ids:
            landmark_name_instance = ""
            for bbox in landmark_name_idx_file['bboxes']:
                if int(bbox['object_id']) == i:
                    landmark_name_instance = bbox['landmark']
                break
            landmark_names_nearest.append(landmark_name_instance)

          landmark_name_object = None
          for bbox in landmark_name_idx_file['bboxes']:
              if int(bbox['object_id']) == item_id:
                  landmark_name_object = bbox['landmark']
              break
          if landmark_name_object is not None:
            landmark_prompt = "The landmark of this object: " + landmark_name_object + ". "

          if any(landmark_names_nearest):
            non_empty_landmark = [item for item in landmark_names_nearest if item]
            landmark_prompt = ", ".join(non_empty_landmark)
            landmark_prompt = ". The landmark next to this object: " + landmark_prompt + ". "
          landmark_prompt = landmark_prompt + semantic_prompt

          # GT
          lang_ans =  []
          for i_instance in cand_instance_ids:
            if (i_instance == object_id) and (i_instance == item_id):
                lang_ans.append(1)
            else:
                lang_ans.append(0)
          lang_gt = lang_ans[0]
          pc_feats = pc_feats[0]
          landmark_feats = landmark_feats[0]

        if scene_name == "UrbanBIS":

          scene_id_converted = scene_id.replace("_", "/", 1)
          feat_file = os.path.join(self.CONF.PATH.UrbanBIS_FEAT, scene_id_converted + "_feat.json")
          bbox_file = os.path.join(self.CONF.PATH.UrbanBIS_BOX, scene_id_converted + "_bbox.json")
          pc_file = os.path.join(self.CONF.PATH.UrbanBIS_Inst, scene_id_converted + ".txt")
          raster_file = os.path.join(self.CONF.PATH.UrbanBIS_MAP, scene_id_converted + ".tif")
          image = safe_read_raster(raster_file)
          global_image = np.transpose(image, (1, 2, 0))

          with open(feat_file, 'r') as json_file:
            feats = json.load(json_file)

          with open(bbox_file, 'r') as json_file:
            bbox = json.load(json_file)

          point_cloud = np.loadtxt(pc_file)

          instance_bboxes = [] # object instance
          instance_bbox = []
          instance_labels = []

          for i in range(len(bbox['bboxes'])):
            bbox_instance = np.array(bbox['bboxes'][i]['bbox'])
            instance_bboxes.append(bbox_instance)
            if bbox['bboxes'][i]['object_id'] ==  object_id:
                instance_bbox = bbox['bboxes'][i]['bbox']
          
          instance_bboxes = np.stack(instance_bboxes)
          instance_labels = instance_bboxes[:, -1]

          if object_id is None:
            pc_feats = np.zeros((1, 1024))
            landmark_feats = np.zeros((1, 768))
            pc_feats_nearest = np.zeros((5, 1024))
            landmark_feats_nearest = np.zeros((5, 768))
            local_image = np.transpose(local_image, (1, 2, 0))

            category_ids = point_cloud[:, 8]
            unique_category_ids = np.unique(category_ids)
            id_to_category = {v: k for k, v in self.Fine_grained_building_category.items()}
            category_names = [id_to_category.get(int(cid), "Ordinary Building") for cid in unique_category_ids]

            if len(category_names) == 1:
              landmark_prompt = f"The scene contains {category_names[0].lower()} buildings."
            else:
              landmark_prompt = "The scene contains " + ", ".join([c.lower() for c in category_names[:-1]]) \
                   + f", and {category_names[-1].lower()} buildings."

            object_prompt = ""

          else: 
            pc_feats = []
            cand_instance_ids = [cand_id for cand_id in np.unique(instance_labels) if cand_id != -100]
            nearest_instance_ids = find_nearest_instances(instance_bboxes, object_id, self.K)

            item_id = object_id

            # Image - Top view feat
            min_x = np.min(point_cloud[:, 0])
            min_y = np.min(point_cloud[:, 1])
            max_x = np.max(point_cloud[:, 0])
            max_y = np.max(point_cloud[:, 1])

            X_pixel = int(instance_bbox[0] - min_x) / 0.1
            Y_pixel = int(instance_bbox[1] - min_y) / 0.1
            X_delta = int(instance_bbox[3]) / 0.1
            Y_delta = int(instance_bbox[4]) / 0.1     
            Image_LH = max(X_delta, Y_delta)

            image_with_rectangle = image[:, :, :]
            x1 = int(X_pixel - int(X_delta))
            y1 = int(image.shape[1] - Y_pixel - int(1.0 * Y_delta))

            x2 = int(X_pixel + int(X_delta))
            y2 = int(image.shape[1] - Y_pixel + int(1.0 * Y_delta))
            thickness = min(6, x2 - x1, y2 - y1)

            if min(x2 - x1, y2 - y1) < 100:
              Image_LH = 10 * max(X_delta, Y_delta)
              thickness = 8
        
            if min(x2 - x1, y2 - y1) > 100:
              Image_LH = 10 * max(X_delta, Y_delta)
              thickness = 20

            for t in range(thickness):
              try:
                image_with_rectangle[0, y1:y2, x1 + t + 1] = 255
                image_with_rectangle[0, y1:y2, x2 - t - 1] = 255
                image_with_rectangle[0, y1 + t + 1, x1:x2] = 255
                image_with_rectangle[0, y2 - t - 1, x1:x2] = 255
                image_with_rectangle[1, y1:y2, x1 + t + 1] = 0
                image_with_rectangle[1, y1:y2, x2 - t - 1] = 0
                image_with_rectangle[1, y1 + t + 1, x1:x2] = 0
                image_with_rectangle[1, y2 - t - 1, x1:x2] = 0
                image_with_rectangle[2, y1:y2, x1 + t + 1] = 0
                image_with_rectangle[2, y1:y2, x2 - t - 1] = 0
                image_with_rectangle[2, y1 + t + 1, x1:x2] = 0
                image_with_rectangle[2, y2 - t - 1, x1:x2] = 0
              except IndexError:
                continue

            if min(x2 - x1, y2 - y1) >= 100:
              local_image = extract_subimage_urbanbis(image_with_rectangle, image.shape[1] - Y_pixel, X_pixel, 10 * Y_delta, 10 * X_delta)
            
            if min(x2 - x1, y2 - y1) < 100:
              local_image = extract_subimage_urbanbis(image_with_rectangle, image.shape[1] - Y_pixel, X_pixel, 20 * Y_delta, 20 * X_delta)

            local_image = np.transpose(local_image, (1, 2, 0))

            pc_feats = []
            landmark_feats = []
            try:
              feat_instance = feats[str(int(item_id))]
            except (KeyError, TypeError):
              feat_instance = np.zeros((1, 1024))
            pc_feats.append(feat_instance)
            pc_feats = np.vstack(pc_feats)

            landmark_feat_instance = np.zeros((1, 768))
            landmark_feats.append(landmark_feat_instance)
            landmark_feats = np.vstack(landmark_feats)
            try:
              while len(nearest_instance_ids) < self.K:
                nearest_instance_ids.append(object_id)
            except AttributeError:
              nearest_instance_ids = [object_id] * self.K

            pc_feats_nearest = []
            for i in nearest_instance_ids:
              feat_instance = np.zeros((1, 1024))
              try:
                feat_instance = feats[str(int(i))]
              except (KeyError, TypeError):
                feat_instance = np.zeros((1, 1024))
              pc_feats_nearest.append(feat_instance)
            pc_feats_nearest = np.vstack(pc_feats_nearest)

            landmark_feats_nearest = []
            for i in nearest_instance_ids:
              feat_instance = np.zeros((1, 768))
              landmark_feats_nearest.append(feat_instance)
            landmark_feats_nearest = np.vstack(landmark_feats_nearest)

            point_cloud_instance = point_cloud[point_cloud[:, 7] == item_id]
            CategoryID = point_cloud_instance[1,8]
            id_to_category = {v: k for k, v in self.Fine_grained_building_category.items()}
            category = id_to_category.get(CategoryID, "Ordinary Building")
            landmark_prompt = " The class of this building object belongs to '" + category.lower() + "'. "
            object_prompt = ""
            bbox = instance_bbox
            x, y, z = bbox[0], bbox[1], bbox[2]
            dx, dy, dz = bbox[3], bbox[4], bbox[5]

            object_prompt = (
              f"The object is located at coordinates "
              f"({x:.2f}, {y:.2f}, {z:.2f}) meters, "
              f"with spatial spans of {dx:.2f} meters along X, "
              f"{dy:.2f} meters along Y, and {dz:.2f} meters along Z."
            )

            lang_gt = 1

        # (Image Processing) Using LLaVA as backbone for City-LLM 
        image_path = images[idx] 
        json_path = jsons[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        image = local_image
        ori_size = image.shape[:2]
        try:
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
            ][0]
        except ZeroDivisionError:
            image_clip = self.clip_image_processor.preprocess(np.zeros((3,3,3)), return_tensors="pt")[
            "pixel_values"
            ][0]

        mask, sents, is_sentence = get_mask_from_json(json_path, global_image)
        if len(sents) >= self.num_classes_per_sample:
          sampled_inds = np.random.choice(
              list(range(len(sents))), size=self.num_classes_per_sample, replace=False
          )
        else:
          sampled_inds = list(range(len(sents)))
        
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [
          (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]

        image = np.zeros((9, 9, 3))
        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        image_name = image_path.split("/")[-1]
        if self.explanatory != -1 and image_name in self.img_to_explanation:
          if random.random() < self.explanatory:
              choice = 2
          else:
              choice = random.randint(0, 1)
        
        global_image_clip = self.clip_image_processor.preprocess(global_image, return_tensors="pt")[
        "pixel_values"
        ][0]   

        ### Language Embedding for SensatUrban and UrbanBIS
        questions = []
        answers = []
        answer_list = []
        question_list = []

        final_question = object_prompt + landmark_prompt + question

        question_list.append(final_question)
        answer_list.append(answer)
        len_question_answer = len(question_list)

        # for text in answer_list:
        for i in range(len_question_answer):
            questions.append(question_list[i]) 
            img_name = image_path.split("/")[-1]
            answers.append(answer_list[i])
            conversations = []
            conv = conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        image_name = image_path.split("/")[-1]
        if (
            self.explanatory != -1
            and image_name in self.img_to_explanation
            and choice == 2
        ):
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label
        else:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            global_image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents, 
            lang_gt,  
            pc_feats,
            landmark_feats,
            pc_feats_nearest,
            landmark_feats_nearest,
        )

def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes, train_scenes_to_use=None, val_scenes_to_use=None):
    # get initial scene list
    if train_scenes_to_use is not None:
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train if data["scene_id"] in train_scenes_to_use])))
    else:
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        
    if val_scenes_to_use is not None:
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val if data["scene_id"] in val_scenes_to_use])))
    else:        
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        
    if num_scenes == -1:
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes
        
    train_scene_list = train_scene_list[:num_scenes]

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    new_scanrefer_val = []
    for data in scanrefer_val:
        if data["scene_id"] in val_scene_list:
            new_scanrefer_val.append(data)

    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list


def one_hot(length, position):
    zeros = [0 for _ in range(length)]
    zeros[position] = 1
    zeros = np.array(zeros)
    return zeros


def shuffle_items_with_indices(lst):
    indices = list(range(len(lst)))  
    random.shuffle(indices) 
    shuffled_items = [lst[idx] for index, idx in enumerate(indices)]
    return shuffled_items, indices


def extract_subimage(image, center_x, center_y, width, height):

    start_x = max(center_x - width // 2, 0)
    end_x = min(center_x + width // 2, image.shape[2])
    start_y = max(center_y - height // 2, 0)
    end_y = min(center_y + height // 2, image.shape[1])
    subimage = image[:,  start_x:end_x,start_y:end_y]
    
    return subimage

def extract_subimage_urbanbis(image, center_x, center_y, width, height):

    start_x = max(center_x - width // 2, 0)
    end_x = min(center_x + width // 2, image.shape[1])
    start_y = max(center_y - height // 2, 0)
    end_y = min(center_y + height // 2, image.shape[2])
    subimage = image[:, int(start_x):int(end_x), int(start_y):int(end_y)]
    
    return subimage

def find_rows_with_value(arr, value):
    rows_with_value = np.where(arr == value)[0]
    return rows_with_value

def find_nearest_instances(arr, objectID, K):
    object_row = arr[arr[:, -1] == objectID]
    if object_row.shape[0] == 0:
        return []

    object_xyz = object_row[:, :3]
    distances = np.sqrt(np.sum((arr[:, :3] - object_xyz) ** 2, axis=1))
    distances[arr[:, -1] == objectID] = np.inf
    nearest_indices = np.argsort(distances)[:K]

    return arr[nearest_indices, -1].tolist()


def safe_read_raster(raster_file):
    try:
        with rasterio.open(raster_file) as src:
            image = src.read()
            if image is None or image.size == 0:
                return np.zeros((3, 224, 224), dtype=np.uint8)
            
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            
            if image.shape[1] == 0 or image.shape[2] == 0:
                return np.zeros((3, 224, 224), dtype=np.uint8)
            
            return image
        
    except Exception as e:
        return np.zeros((3, 224, 224), dtype=np.uint8)
