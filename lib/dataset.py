import os
import sys
import torch
import random
import numpy as np
# import rasterio
# from rasterio.plot import show
from skimage.transform import resize
import imageio
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from torch.utils.data import Dataset
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.sensaturban.model_util_sensaturban import rotate_aligned_boxes_along_axis
# from torchsparse import SparseTensor
# from torchsparse.utils.quantize import sparse_quantize
# from torchsparse.utils.collate import sparse_collate_fn

torch.multiprocessing.set_sharing_strategy('file_system')

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
    # input image: [channel,length,width]

    start_x = max(center_x - width // 2, 0)
    end_x = min(center_x + width // 2, image.shape[2])
    start_y = max(center_y - height // 2, 0)
    end_y = min(center_y + height // 2, image.shape[1])
    subimage = image[:, start_x:end_x, start_y:end_y]
    
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

class ReferenceDataset(Dataset):
    def __init__(self, 
                 args,
                DC, CONF, dataset, scanrefer, scanrefer_all_scene,
                split="train", # split_2 in city_vg_dataset
                num_points=40000,
                use_height=False,
                use_color=False,
                augment=False,
                use_cache=False,
        ):
        self.args = args
        self.DC = DC        
        self.CONF = CONF
        self.use_cache = use_cache
        self.cache = {}
        self.num_cands = args.num_cands
        
        self.dataset = dataset
        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene  # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment if split == "train" else False
        self.use_landmark = args.use_landmark
        self.num_inst_points = self.args.num_inst_points

        if self.dataset == 'sensaturban':
            self.other_object_cat = -1
            self._load_sensaturban_data()
        else:
            raise NotImplementedError

        self.voxel_size_ap = self.args.voxel_size_ap
        self.voxel_size_glp = self.args.voxel_size_glp

    def __len__(self):
        #print(len(self.scanrefer))
        return len(self.scanrefer)

    def __getitem__(self, idx):

        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = self.scanrefer[idx]["object_name"]
        ann_id = int(self.scanrefer[idx]["ann_id"])
        object_cat = self.raw2label[object_name] if object_name in self.raw2label else self.other_object_cat
        assert object_cat >= 0
        assert object_id >= 0

        MAX_NUM_OBJ = self.args.max_num_object
        MAX_NUM_LANDMARK = self.args.max_num_landmark
        query = self.scanrefer[idx]["description"]
        subimage=np.zeros((3,9,9))

        # Use cache
        #print(self.use_cache) #true
        if self.use_cache and scene_id in self.cache:
            mesh_vertices, instance_labels, semantic_labels, instance_bboxes, landmark_names, landmark_ids, globalShift = self.cache[scene_id]
        else:
            pg_file = os.path.join(CONF.PATH.SCAN_DATA, scene_id+".pth")
            coords, colors, label_ids, instance_ids, label_ids_pg, instance_ids_pg, instance_bboxes, \
                landmark_names, landmark_ids, globalShift = torch.load(pg_file)

            mesh_vertices = np.concatenate([coords, colors], axis=1)
            #print(mesh_vertices.shape)
            instance_bboxes = np.stack([instance_bboxes[instance_id] for instance_id in sorted(instance_bboxes.keys()) if instance_id != -100])

            raster_file = os.path.join(CONF.PATH.SCAN_DATA, scene_id+".tif")
            with rasterio.open(raster_file) as src:
                image = src.read()
            image=(image/127.5)-1 # [channel,length,width]
                  
            #instance_bboxes_pixel=instance_bboxes[:,0:6]
            instance_bboxes_pixel = np.copy(instance_bboxes)
            instance_bboxes_pixel[:,0]=instance_bboxes_pixel[:,0]/0.1 #X的步长
            instance_bboxes_pixel[:,1]=instance_bboxes_pixel[:,1]/0.1 #Y的步长 
            instance_bboxes_pixel[:,3]=instance_bboxes_pixel[:,3]/0.1 #长度
            instance_bboxes_pixel[:,4]=instance_bboxes_pixel[:,4]/0.1 #宽度
            arr = instance_bboxes[:,-1]
            row = find_rows_with_value(arr, object_id)
            if len(row) == 0:
                subimage = np.zeros((256,256,3))
            else:         
                X_pixel=int(instance_bboxes_pixel[row,1])
                Y_pixel=int(instance_bboxes_pixel[row,2])
                X_delta=int(instance_bboxes_pixel[row,4])
                Y_delta=int(instance_bboxes_pixel[row,5])
                Image_LH = max(X_delta,Y_delta)
                subimage = extract_subimage(image, Y_pixel, X_pixel, 6*Image_LH, 6*Image_LH)

            if self.args.no_gt_instance:
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
            point_cloud[:, 3:6] = point_cloud[:, 3:6] / 127.5 - 1 # same as sensaturban

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)
            
        if self.num_points > 0:
            point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
            instance_labels = instance_labels[choices]
            semantic_labels = semantic_labels[choices]

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        
        ref_center_label = np.zeros(3)  # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3)  # bbox size residual for reference target
        scene_points = np.zeros((1, 10))
        
        #if self.split != "test":
        #print(self.split) #train
        if self.split != "gt":            
            num_bbox = len(instance_bboxes) if len(instance_bboxes) < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox, :] = instance_bboxes[:MAX_NUM_OBJ, 0:6]
            # NOTE: set size class as semantic class. Consider use size2class.
            class_ind = [self.DC.label_id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - self.DC.mean_size_arr[class_ind, :]

            # construct the reference target label for each bbox
            for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]):
                if gt_id == object_id:
                    ref_center_label = target_bboxes[i, 0:3]
                    ref_heading_class_label = angle_classes[i]
                    ref_heading_residual_label = angle_residuals[i]
                    ref_size_class_label = size_classes[i]
                    ref_size_residual_label = size_residuals[i]
        else:
            num_bbox = 1
        
        instance_class = []
        pred_obbs = []
        geo_feats = []

        cand_instance_ids = [cand_id for cand_id in np.unique(instance_labels) if cand_id != -100]

        nearest_instance_ids = find_nearest_instances(instance_bboxes, object_id,5) # nearest instance finding
        nearest_instance_ids = np.insert(nearest_instance_ids, 0, object_id)

        if self.num_cands > 0:
            if object_id in cand_instance_ids:
                cand_instance_ids.remove(object_id)    
                cand_instance_ids = sorted([object_id] + random.sample(cand_instance_ids, min(self.num_cands - 1, len(cand_instance_ids))))
            else:
                cand_instance_ids = sorted(random.sample(cand_instance_ids, min(self.num_cands, len(cand_instance_ids))))

        for i_instance in cand_instance_ids:

            ind = np.nonzero(instance_labels == i_instance)[0]
            ins_class = semantic_labels[ind[0]]

            if ins_class in self.DC.label_id2class:
                x = point_cloud[ind]
                ins_class = self.DC.label_id2class[int(ins_class)]
                instance_class.append(ins_class)
                pc = x[:, :3]
                center = 0.5 * (pc.min(0) + pc.max(0))
                size = pc.max(0) - pc.min(0)
                ins_obb = np.concatenate((center, size, np.array([0])))

                if self.num_inst_points > 0:
                    x = random_sampling(x, self.num_inst_points)


                if ins_class == object_cat:
                    pc = x[:, :3]

                    ins_coords, ins_indices = sparse_quantize(pc, self.voxel_size_ap, return_index=True)
                    ins_coords = torch.tensor(ins_coords, dtype=torch.int)
                    ins_feats = torch.tensor(x[ins_indices], dtype=torch.float16) #float
                    geo_feat = SparseTensor(coords=ins_coords, feats=ins_feats)


                    if len(ins_obb) < 2:
                        continue

                    pred_obbs.append(ins_obb)
                    geo_feats.append(geo_feat)

        if len(geo_feats) > MAX_NUM_OBJ:
            geo_len = MAX_NUM_OBJ
            geo_feats = geo_feats[:MAX_NUM_OBJ]
            pred_obbs = pred_obbs[:MAX_NUM_OBJ]
        elif len(geo_feats) < MAX_NUM_OBJ:
            geo_len = len(geo_feats)
        else:
            geo_len = MAX_NUM_OBJ

        #
        # Landmark
        #
        landmark_obbs = []
        landmark_feats = []
        landmark_texts = []
        landmark_len = None
        
        local_instance_class = []
        local_pred_obbs = []
        local_geo_feats = []

        for i_instance in nearest_instance_ids:
            ind = np.nonzero(instance_labels == i_instance)[0]
            ins_class = semantic_labels[ind[0]]
            if ins_class in self.DC.label_id2class:
                x = point_cloud[ind]
                ins_class = self.DC.label_id2class[int(ins_class)]
                local_instance_class.append(ins_class)

                pc = x[:, :3]
                center = 0.5 * (pc.min(0) + pc.max(0))
                size = pc.max(0) - pc.min(0)
                ins_obb = np.concatenate((center, size, np.array([0])))
                if self.num_inst_points > 0:
                    x = random_sampling(x, self.num_inst_points)

                if ins_class == object_cat:
                    pc = x[:, :3]
                    ins_coords, ins_indices = sparse_quantize(pc, self.voxel_size_ap, return_index=True)
                    ins_coords = torch.tensor(ins_coords, dtype=torch.int)
                    ins_feats = torch.tensor(x[ins_indices], dtype=torch.float)
                    geo_feat = SparseTensor(coords=ins_coords, feats=ins_feats)

                    if len(ins_obb) < 2:
                        continue

                    local_pred_obbs.append(ins_obb)
                    local_geo_feats.append(geo_feat)

        if len(local_geo_feats) > 6:
            local_geo_len = 6
            local_geo_feats = local_geo_feats[:6]
            local_pred_obbs = local_pred_obbs[:6]
        elif len(local_geo_feats) < 6:
            local_geo_len = len(local_geo_feats)
        else:
            local_geo_len = 6

        if self.use_landmark:
            for i_landmark in sorted(np.unique(landmark_ids)):
                if i_landmark == -100:
                    continue
                ind = np.nonzero(landmark_ids == i_landmark)[0]
                x = point_cloud[ind]

                pc = x[:, :3]
                center = 0.5 * (pc.min(0) + pc.max(0))
                size = pc.max(0) - pc.min(0)
                landmark_obb = np.concatenate((center, size, np.array([0])))
                landmark_obbs.append(landmark_obb)
                if self.num_inst_points > 0:
                    x = random_sampling(x, self.num_inst_points)

                pc = x[:, :3]
                ins_coords, ins_indices = sparse_quantize(pc, self.voxel_size_ap, return_index=True)
                ins_coords = torch.tensor(ins_coords, dtype=torch.int)
                ins_feats = torch.tensor(x[ins_indices], dtype=torch.float)
                landmark_feat = SparseTensor(coords=ins_coords, feats=ins_feats)
                landmark_feats.append(landmark_feat)
                landmark_texts.append(landmark_names[i_landmark])

            if len(landmark_feats) > MAX_NUM_LANDMARK:
                landmark_len = MAX_NUM_LANDMARK
                landmark_feats = landmark_feats[:MAX_NUM_LANDMARK]
                landmark_texts = landmark_texts[:MAX_NUM_LANDMARK]
            elif len(landmark_feats) < MAX_NUM_LANDMARK:
                landmark_len = len(landmark_feats)
            else:
                landmark_len = MAX_NUM_LANDMARK        
                

        data_dict = {}
        data_dict["istrain"] = np.array(1) if self.split == "train" else np.array(0)        
        data_dict['geo_feats'] = geo_feats
        data_dict['geo_len'] = np.array(geo_len).astype(np.int64)
        data_dict['pred_obb_batch'] = pred_obbs
        data_dict["query"] = query 
        data_dict['point_min'] = point_cloud.min(0)[:3]
        data_dict['point_max'] = point_cloud.max(0)[:3]
        data_dict['instance_class'] = instance_class
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)
        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["ann_id"] = np.array(ann_id).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)

        img_transposed = np.transpose(subimage, (1, 2, 0))
        resized_img_transposed = resize(img_transposed, (1024, 1024, 3), anti_aliasing=True) # size of vis_feat extraction
        resized_img = np.transpose(resized_img_transposed, (2, 0, 1))     
        data_dict["vis_feat"] = resized_img # added data_dict for vis_feat

        data_dict['local_geo_feats'] = local_geo_feats
        data_dict['local_pred_obb_batch'] = local_pred_obbs
        data_dict['local_geo_len'] = np.array(local_geo_len).astype(np.int64)
        
        if self.use_landmark:
            data_dict['landmark_feats'] = landmark_feats
            data_dict["landmark_texts"] = landmark_texts 
            data_dict['landmark_len'] = np.array(landmark_len).astype(np.int64)
        data_dict["unique_multiple"] = np.array(
            self.unique_multiple_lookup[scene_id][object_id][ann_id]).astype(np.int64)

        return data_dict
    

    def _get_raw2label(self, SCANNET_V2_TSV):
        # Mapping
        scannet_labels = DC.type2class.keys()
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
                except KeyError:
                    all_sem_labels[scene_id].append(self.other_object_cat)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = int(data["object_id"])
            object_name = data["object_name"] # Ground, Building, etc
            ann_id = int(data["ann_id"])

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
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
        #print(self.scanrefer)

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

    @staticmethod
    def collate_fn(batch):
        outputs = sparse_collate_fn(batch)
        pred_obb_batch = []
        for input in batch:
            if not len(input['pred_obb_batch']) < 2:
                pass
            pred_obb_batch.append(np.array(input['pred_obb_batch']))
            
        outputs['pred_obb_batch'] = pred_obb_batch

        return outputs
