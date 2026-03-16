from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer for grounding token
        in_dim = config.hidden_size
        out_dim = config.out_dim

        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        
        text_fc_vg = [
            nn.Linear(out_dim, 1),
        ]
        self.text_hidden_fcs_vg = nn.ModuleList([nn.Sequential(*text_fc_vg)])
        self.text_hidden_fcs_vg.train()
        for param in self.text_hidden_fcs_vg.parameters():
            param.requires_grad = True



class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        global_images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        lang_ans_list: List[torch.FloatTensor],
        pc_feats: torch.FloatTensor,
        landmark_feats: torch.FloatTensor,
        nearest_pc_feats: torch.FloatTensor,
        nearest_landmark_feats: torch.FloatTensor,
        inference: bool = False,
        **kwargs,
    ):
        
        # print(scene_pc_feats.shape)
        # print(scene_code.shape)

        image_embeddings = self.get_visual_embs(images) # no need?
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx

        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )

        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 255 + 1 + 1 + 5 + 5 + 256)).bool().cuda(),
            ],
            dim=1,
        ) # only pc feats

        ### Token mask in ablation study
        '''
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255 + 800 + 1 + 1 + 5 + 5)).bool().cuda(), seg_token_mask], 
            dim=1,
        ) # image, pc and landmark feats (+2*K)
        '''

        ### Token mask in ablation study
        '''
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 5)).bool().cuda(), seg_token_mask], 
            dim=1,
        ) # only pc feats (including Nei.)
        '''
        
        '''
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255 + 800 + 256 + 1 + 1 + 5 + 5)).bool().cuda(), seg_token_mask], 
            dim=1,
        ) # 
        '''

        '''
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255 + 1 + 1 + 5 + 5)).bool().cuda(), seg_token_mask], 
            dim=1,
        ) # 
        '''
        
        ### Forward in testing
        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list # output of llava
            output = None

        else: 
            # for local
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            # for global
            global_images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                global_images_clip_i = (
                    global_images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                global_images_clip_list.append(global_images_clip_i)
            global_images_clip = torch.cat(global_images_clip_list, dim=0)
            
            # Create <Source> as carrier for multimodal inputs
            Source = []
            Source.append(images_clip) # enter llava
            # Source.append(global_images_clip)
            Source.append(pc_feats)
            Source.append(nearest_pc_feats)
            Source.append(landmark_feats)
            Source.append(nearest_landmark_feats)
            Source.append(global_images_clip)

            '''
            print("#########3")
            print(images_clip.shape)
            print(pc_feats.shape)
            print(nearest_pc_feats.shape)
            print(landmark_feats.shape)
            print(nearest_landmark_feats.shape)
            print(global_images_clip.shape)
            print("################")
            print(images_clip[:1].shape)
            print(pc_feats[:1].shape)
            print(nearest_pc_feats[:1].shape)
            print(landmark_feats[:1].shape)
            print(nearest_landmark_feats[:1].shape)
            print(global_images_clip[:1].shape)
            print(r)
            '''

            # Follow LLaVA input -> multi-modality source
            output = super().forward(
                images=Source,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )

            output_hidden_states = output.hidden_states # Respons for original multi-modality input 

        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        model_output = output
        output = model_output.logits
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight

        # loss = loss_BCE_VG + ce_loss
        loss = ce_loss # only chat

        return {
            "loss": loss,
            "ce_loss": ce_loss,
        }

    '''
    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        global_images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        lang_ans_list: List[torch.FloatTensor],
        pc_feats: torch.FloatTensor,
        landmark_feats: torch.FloatTensor,
        nearest_pc_feats: torch.FloatTensor,
        nearest_landmark_feats: torch.FloatTensor,
        scene_pc_feats: torch.FloatTensor,
        scene_code: torch.FloatTensor,
        inference: bool = False,
        **kwargs,
    ):
    '''
    def evaluate(
        self,
        pc_feats,
        landmark_feats,
        nearest_pc_feats,
        nearest_landmark_feats,
        images_clip,
        images,
        input_ids,
        global_images_clip,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):       

        with torch.no_grad():
            
            pc_feats = pc_feats[None, :]
            landmark_feats = landmark_feats[None, :]

            Source = []
            '''
            Source.append(images_clip)
            Source.append(pc_feats)
            Source.append(nearest_pc_feats)
            Source.append(landmark_feats)
            Source.append(nearest_landmark_feats)
            Source.append(global_images_clip)
            Source.append(scene_pc_feats)
            Source.append(scene_code)
            '''

            Source.append(images_clip)
            Source.append(pc_feats)
            Source.append(nearest_pc_feats)
            Source.append(landmark_feats)
            Source.append(nearest_landmark_feats)
            Source.append(global_images_clip)

            outputs = self.generate(
                images=Source,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                top_k=30,
                top_p=0.9,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                num_return_sequences=1,
                temperature=1.0,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            # print(images_clip.shape)
            # print(input_ids)
            # print(input_ids.shape)
            # print(output_ids)
            # print(output_ids.shape)

        return output_ids
