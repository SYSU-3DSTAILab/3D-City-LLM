import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.CityLLM import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

from lib.config import CONF
from data.sensaturban.model_util_sensaturban import SensatUrbanDatasetConfig
import open3d as o3d
# import gradio as gr
import plotly.graph_objects as go
import json
import rasterio
import time
import re
from PIL import Image


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="./LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--max_num", default=400, type=int)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    for i in range(args.max_num):
        added_token = "obj" + "<" + str(i) + ">"
        num_added_tokens = tokenizer.add_tokens(added_token)

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()
    DC = SensatUrbanDatasetConfig()            
    print("preparing data...")

    SCANREFER_TRAIN = json.load(open(os.path.join("/content/drive/MyDrive/City-LLM/CityLLMData/Total_Meta_Data_Train_Val_Test/Object_Caption/SensatUrban_Object_Caption_Part_2_Test_sampled.json"))) # TODO: change this

    SCANREFER_VAL = json.load(open("/content/drive/MyDrive/City-LLM/CityLLMData/Total_Meta_Data_Train_Val_Test/Object_Caption/SensatUrban_Object_Caption_Part_2_Test_sampled.json")) # TODO: change this

    file_path = "/content/drive/MyDrive/City-LLM/TestResults/City-LLM/CityLLM_SensatUrban_Object_Caption_Part2_Results_v1.txt" # TODO: change this

    scanrefer_train, scanrefer_val, all_scene_list= get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, -1)
    scanrefer = {"train": scanrefer_train, "val": scanrefer_val}
    scanrefer = scanrefer["val"]
    output_dict = []
    K = 5
    Fine_grained_building_category = {'Commercial': 0, 'Residential': 1, 'Office': 2, 'Cultural': 3, 'Transportation': 4, 'Municipal': 5, 'Temporary': 6, 'Unclassified': 7}
    print("The total val sample is: ",len(scanrefer))

    for i in range(len(scanrefer)):

        start_time = time.time()

        idx = i
        print(idx)
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        scene_inf = scanrefer[idx]["scene_id"]
        obj_id = int(scanrefer[idx]["object_id"])
        task_type = scanrefer[idx]["type"]
        question = scanrefer[idx]["question"]
        GT_answer = scanrefer[idx]["answer"]

        if obj_id != "-":
          object_id = int(obj_id)
        else:
          object_id = None

        object_prompt = ""
        landmark_prompt = ""

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

          pg_file = os.path.join(CONF.PATH.SCAN_DATA, scene_id+".pth")
          feat_file = os.path.join("/content/drive/MyDrive/City-LLM/SensatUrban_Feat", scene_id+".json")
          landmark_file = os.path.join("/content/drive/MyDrive/LISA_Grounding/data/sensaturban/landmark_feat_128_all", scene_id+".landmark_feat.json")
          landmark_nearest_file = os.path.join("/content/drive/MyDrive/LISA_Grounding/data/sensaturban/landmark_feat_128_all", scene_id+".landmark_feat.json")
          landmark_name_file = os.path.join("/content/drive/MyDrive/Cityrefer_data/box", scene_id+"_bbox.json")

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

          instance_labels = instance_ids_pg
          semantic_labels = label_ids_pg

          point_cloud = mesh_vertices[:, 0:6]
          point_cloud[:, 3:6] = point_cloud[:, 3:6] / 127.5 - 1 

          floor_height = np.percentile(point_cloud[:, 2], 0.99)
          height = point_cloud[:, 2] - floor_height
          point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

          instance_class = []
          pc_feats = []

          cand_instance_ids = [cand_id for cand_id in np.unique(instance_labels) if cand_id != -100]
          nearest_instance_ids = find_nearest_instances(instance_bboxes, object_id, 5)
          nearest_instance_landmark_ids = find_nearest_instances(instance_bboxes, object_id, 5)
        
          item_id = object_id
          raster_file = os.path.join("/content/drive/MyDrive/Cityrefer_data/RGB_map", scene_id+".tif")
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

            Image_LH = max(X_delta, Y_delta)

            image_with_rectangle = image[:, :, :]
            x1 = int(image.shape[2] - Y_pixel - int(Y_delta/2))
            y1 = int(X_pixel - int(X_delta/2))
            x2 = int(image.shape[2] - Y_pixel + int(Y_delta/2))
            y2 = int(X_pixel + int(X_delta/2))
            thickness = min(10, x2 - x1, y2 - y1)

            if min(10, x2 - x1, y2 - y1) < 100:
              Image_LH = 10 * max(X_delta, Y_delta)
              thickness = 2
            
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

          while len(nearest_instance_ids) < 5:
            nearest_instance_ids.append(object_id)

          if isinstance(nearest_instance_landmark_ids, np.ndarray):
            nearest_instance_landmark_ids = nearest_instance_landmark_ids.tolist()

          while len(nearest_instance_landmark_ids) < 5:
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

          lang_ans =  []
          for i_instance in cand_instance_ids:
            if (i_instance == object_id) and (i_instance == item_id):
                lang_ans.append(1)
            else:
                lang_ans.append(0)
          lang_gt = 1
          pc_feats = pc_feats[0]
          landmark_feats = landmark_feats[0]

        if scene_name == "UrbanBIS":

          scene_id_converted = scene_id.replace("_", "/", 1)
          feat_file = os.path.join("/content/drive/MyDrive/UrbanBIS_data/UrbanBIS_feat", scene_id_converted + "_feat.json")
          bbox_file = os.path.join("/content/drive/MyDrive/UrbanBIS_data/UrbanBIS_bbox", scene_id_converted + "_bbox.json")
          pc_file = os.path.join("/content/drive/MyDrive/UrbanBIS_data/UrbanBIS_Inst", scene_id_converted + ".txt")
          raster_file = os.path.join("/content/drive/MyDrive/UrbanBIS_data/UrbanBIS_map", scene_id_converted + ".tif")
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
            id_to_category = {v: k for k, v in Fine_grained_building_category.items()}
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
            nearest_instance_ids = find_nearest_instances(instance_bboxes, object_id, 5)
            item_id = object_id

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
              while len(nearest_instance_ids) < K:
                nearest_instance_ids.append(object_id)
            except AttributeError:
              nearest_instance_ids = [object_id] * K

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
            id_to_category = {v: k for k, v in Fine_grained_building_category.items()}
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

        local_image_np = local_image
        original_size_list = [local_image_np.shape[:2]]

        try:
            image_clip = clip_image_processor.preprocess(local_image_np, return_tensors="pt")[
            "pixel_values"
            ][0]
        except (ZeroDivisionError, ValueError):
            image_clip = clip_image_processor.preprocess(np.zeros((3,3,3)), return_tensors="pt")[
            "pixel_values"
            ][0]

        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        try:
            image = transform.apply_image(local_image_np)
        except (ValueError, ZeroDivisionError):
            image = transform.apply_image(np.zeros((3,3,3)))

        resize_list = [image.shape[:2]]
        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        global_image_np = global_image
        original_size_list = [local_image_np.shape[:2]]

        try:
            global_image_clip = clip_image_processor.preprocess(global_image_np, return_tensors="pt")[
            "pixel_values"
            ][0]
        except (ZeroDivisionError, ValueError):
            global_image_clip = clip_image_processor.preprocess(np.zeros((3,3,3)), return_tensors="pt")[
            "pixel_values"
            ][0]

        prompt = object_prompt + landmark_prompt + question
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        output_ids = model.evaluate(
            pc_feats,
            landmark_feats,
            pc_feats_nearest,
            landmark_feats_nearest,
            image_clip,
            image,
            input_ids,
            global_image_clip,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )

        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")

        match = re.search(r'ASSISTANT: (.*?)</s>', text_output)
        if match:
            description_output = match.group(1)
            print("Order: " + str(idx) + ",scene_id: " + scene_id + ", object_id: " + str(object_id))
            print("Pred: " + description_output)
            print("GT: " + GT_answer)
        
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write("Order: " + str(idx) + ",scene_id: " + scene_id + ", object_id: " + str(object_id) + "\n")
            file.write("Pred: " + description_output + "\n")
            file.write("GT: " + GT_answer + "\n")

    print("Completed")


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
        
    # slice train_scene_list
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

def find_nearest_instances(arr, objectID,K):
    object_row = arr[arr[:, -1] == objectID]
    if object_row.shape[0] == 0:
        return []
    object_xyz = object_row[:, :3]
    distances = np.sqrt(np.sum((arr[:, :3] - object_xyz) ** 2, axis=1))
    distances[arr[:, -1] == objectID] = np.inf
    nearest_indices = np.argsort(distances)[:K]
    return arr[nearest_indices, -1]


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

if __name__ == "__main__":
    main(sys.argv[1:])
