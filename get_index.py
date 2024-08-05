import os
import h5py
from RESNET import ResNet
from annoy import AnnoyIndex
from tqdm import tqdm


def get_index_by_VGG(features, img_list):
    h5f = h5py.File('static/index/ResNet_features.h5', 'w')
    h5f.create_dataset('features', data=features)
    h5f.create_dataset('img_list', data=img_list)
    h5f.close()


def get_index_by_VGG_and_Annoy(features):
    annoy_index = AnnoyIndex(512, metric='angular')
    for i in range(len(features)):
        annoy_index.add_item(i, features[i])
    annoy_index.build(10)
    annoy_index.save('static/index/ResNet_Annoy.ann')


if __name__ == '__main__':
    # 可读性一坨但性能优秀的嵌套列表推导式，如果难以理解请看下面实现相同功能的注释代码
    img_list = [os.path.join(root, file) for root, dirs, files in os.walk('static/dataset') for file in files]
    # img_list = []
    # for root, dirs, files in os.walk('static/dataset'):
    #     for file in files:
    #         img_list.append(os.path.join(root, file))
    model = ResNet()
    features = [model.get_feat(img_path) for img_path in tqdm(img_list)]

    get_index_by_VGG(features, img_list)
    get_index_by_VGG_and_Annoy(features)



