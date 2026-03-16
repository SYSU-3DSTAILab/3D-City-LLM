# 3DCity-LLM: Empowering Multi-modality Large Language Models for 3D Perception and Understanding

This is the official PyTorch implementation of 3DCity-LLM. 

## Abstract

 ![overview](fig/City-LLM-Method.png)

<p align="justify">
While Multi-modality Large Language Models (MLLMs) excel in object-centric or indoor scenarios, scaling them to 3D city-scale environments remains a formidable challenge. To bridge this gap, we propose 3DCity-LLM, a unified framework designed for 3D city-scale vision-language perception and understanding. 3DCity-LLM introduces a coarse-to-fine feature encoding strategy that integrates object-level geometry, inter-object relationships, and global scene semantics into a shared linguistic embedding space. To support large-scale training, we present 3DCity-LLM-1.2M dataset that comprises approximately 1.2 million high-quality samples across seven representative task categories, ranging from fine-grained object analysis to complex scene planning. This strictly quality-controlled dataset integrates explicit 3D numerical information and diverse user-oriented simulations to enrich QA diversity and ensure the realism of urban scenarios. Furthermore, we apply a multi-dimensional protocol based on text-similarity metrics and LLM-based semantic assessment to ensure a faithful and comprehensive evaluations. Extensive experiments on two benchmarks demonstrate that 3DCity-LLM significantly outperforms existing state-of-the-art methods, representing a meaningful step toward spatial reasoning and urban intelligence.
</p>

## 💾 Dataset Download

The 3DCity-LLM-1.2M dataset is available at [meta data](https://drive.google.com/drive/folders/1EMSvh1AYympK5Xqj1FmAH-9isTMHYZvT?usp=drive_link). In addition, we have provided all the prepared data in [Google Drive](https://drive.google.com/drive/folders/1EMSvh1AYympK5Xqj1FmAH-9isTMHYZvT?usp=drive_link). Please download the files and place them in the ``.\data`` directory, then change the corresponding path in ``.\lib\config.py``, ``.\utils\cityllm_dataset.py`` and ``.\Test_CityLLM.py``. 

 ![overview](fig/City-LLM-Data.png)

## 📑 Schedule

To facilitate related research, we plan to make 3DCity-LLM open source, including but not limited to the following:

- [x] Release the 3DCity-LLM-1.2M dataset.
- [x] Release the training scripts.
- [x] Provide the weights of **pre-trained 3DCity-LLM model (7B)**.
- [x] Release the dataset construction pipline and dataloader script for chatting on your own city-scale dataset.
- [ ] Release the evaluation scripts.
- [ ] Create an easy-to-use demo.


## 💻 Requirements
The code has been tested on:
- Ubuntu 20.04
- CUDA 12.2
- Python 3.10
- Pytorch 2.1.0
- NVIDIA A100 GPU (80G).

## 🔧 Installation
  
- Create and activate the conda environment
  ```
  conda create -n CityLLM python=3.10
  conda activate CityLLM
  ```

- Install the necessary packages
  ```
  pip install -r requirements.txt
  pip install openai==0.28
  pip install transformers==4.31.0
  pip install deepspeed==0.15.1
  ```

## 🚅 Train
You can train the 3DCity-LLM with pre-trained LLM backbone. It takes only about **24 hours** for training. Note that SAM is available at [backbone model](https://drive.google.com/file/d/1shdmFrhzX3QD-8nhF8OV_zkCGWTZU984/view?usp=drive_link).
```
deepspeed --master_port=24999 Train_CityLLM.py \
  --dataset_dir='./dataset' \
  --vision_pretrained="./sam_vit_h_4b8939.pth" \
  --dataset="cityllm" \
  --sample_rates="1" \
  --exp_name="CityLLM_Train_Model_on_CityLLM_Dataset_v1" \
  --epochs=6 \
  --steps_per_epoch=200 \
  --reason_seg_data='SensatUrban-LISA-EX|train'\
  --explanatory=-1 \
  --no_eval
```
When training process is finished, you should get the full model weight:
```
cd ./runs/CityLLM_Train_Model_on_CityLLM_Dataset_v1/ckpt_model
python zero_to_fp32.py . ../pytorch_model.bin
```
Then, you need to merge the LoRA weights in "pytorch_model.bin", and save the final City-LLM model into your desired path in the Hugging Face format:
```
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_BASED_MODEL" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"
```
We typically use the `xinlai/LISA-7B-v1`(Recommended) or `LLaVA-Lightning-7B-v1-1` as based model (pre-trained with image-text pairs) to be merged. If possible, please pre-train 3DCity-LLM model on the caption tasks in 3DCity-LLM-1.2M dataset, then fine-tune 3DCity-LLM model on the full 3DCity-LLM-1.2M dataset.

## ✏️ Evaluation
You can evaluate the performance on 3DCity-LLM dataset. Please put the PTH model in folder `/path/to/checkpoints` and use the following commands:
```
CUDA_VISIBLE_DEVICES=0 python Test_CityLLM.py \
  --version="PATH_TO_CityLLM_MODEL"
```
The PTH model is available at [3D_CityLLM_Model](https://drive.google.com/drive/folders/1XAPTi_bzebFkdCXrobWcQi_icvxrmgmE?usp=sharing).

## 🤝 Acknowledgement
3DCity-LLM is built upon the wonderful [LLaVA](https://github.com/haotian-liu/LLaVA), [Uni3D](https://github.com/baaivision/Uni3D), [SensatUrban](https://github.com/QingyongHu/SensatUrban), [CityRefer](https://github.com/ATR-DBI/CityRefer) and [UrbanBIS](https://vcc.tech/UrbanBIS). 

## Contact us
For any questions, please contact us via lijp57@whu.edu.cn.
