from semantic_kitti import *
import yaml
# from torchpack.utils.config import configs


# 读取 YAML 文件
with open('/l/users/junchen.liu/PointDR/configs/kitti_syn_poss2stf/default.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

# 访问配置项
dataset_name = config_data['src_dataset1']['name']
dataset_root = config_data['src_dataset1']['root']
num_points = config_data['src_dataset1']['num_points']
voxel_size = config_data['src_dataset1']['voxel_size']


def get_kitti(phase):
    # dataset_config = configs.src_dataset1 if phase == 'train' else configs.tgt_dataset
    dataset = SemanticKITTI(root=dataset_root,
                            num_points=num_points,
                            voxel_size=voxel_size)
    return dataset[phase]

kitti = get_kitti('train')
# print(kitti.reverse_label_name_mapping)  
# {'car': 0, 'bicycle': 1, 'person': 2, 'rider': 3, 'other-ground': 4, 'building': 5, 'fence': 6, 'vegetation': 7, 'trunk': 8, 'pole': 9, 'traffic-sign': 10}

# print(kitti.label_map)
# [255. 255.   0.   0.   0.   0.   0.   0.   0.   0.   0.   1.   0. 255.
#    0. 255. 255.   0. 255.   0. 255.   0.   0.   0.   0.   0.   0.   0.
#    0.   0.   2.   3.   3.   0.   0.   0.   0.   0.   0.   0. 255.   0.
#    0.   0. 255.   0.   0.   0. 255.   4.   5.   6. 255.   0.   0.   0.
#    0.   0.   0.   0. 255.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    7.   8. 255.   0.   0.   0.   0.   0.   0.   0.   9.  10.   0.   0.
#    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    0. 255.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#    0. 255.   2. 255. 255. 255. 255. 255.]