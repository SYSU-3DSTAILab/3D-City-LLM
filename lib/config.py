import os
import sys
from easydict import EasyDict

# path
CONF = EasyDict()
CONF.PATH = EasyDict()

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# SensatUrban data
CONF.PATH.SensatUrban_PG = "./data/SensatUrban/PG"   # TODO: change this
CONF.PATH.SensatUrban_FEAT = "./data/SensatUrban/Uni3D_Feat"   # TODO: change this
CONF.PATH.SensatUrban_LANDMARK = "./data/SensatUrban/BERT_Feat"   # TODO: change this
CONF.PATH.SensatUrban_BOX = "./data/SensatUrban/box" # TODO: change this
CONF.PATH.SensatUrban_MAP = "./data/SensatUrban/RGB_Map" # TODO: change this

# UrbanBIS data 
CONF.PATH.UrbanBIS_FEAT = "./data/UrbanBIS/Uni3D_Feat"   # TODO: change this
CONF.PATH.UrbanBIS_BOX = "./data/UrbanBIS/box" # TODO: change this
CONF.PATH.UrbanBIS_MAP = "./data/UrbanBIS/RGB_Map" # TODO: change this
CONF.PATH.UrbanBIS_Inst = "./data/UrbanBIS/Inst" # TODO: change this