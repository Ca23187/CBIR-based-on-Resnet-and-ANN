import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from RESNET import ResNet
from annoy import AnnoyIndex
import time
import h5py
import shutil
import numpy as np

# 清空并新建tmp文件夹
try:
    shutil.rmtree('static/temp')
except:
    pass
finally:
    os.makedirs('static/temp')

# 特征提取模型
model = ResNet()

# 用于VGG暴力搜索的模型
h5f = h5py.File("static/index/ResNet_features.h5", 'r')
features = h5f['features'][:]
img_list = h5f['img_list'][:]
h5f.close()

# 基于Annoy的模型
annoy_index = AnnoyIndex(512, 'angular')
annoy_index.load('static/index/ResNet_Annoy.ann')

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        img_path = 'static/temp/' + filename
        file.save(img_path)

        query_vector = model.get_feat(img_path)
        start_time = time.time()

        if request.form['algorithm'] == '1':
            scores = np.dot(query_vector, features.T)
            rank_ID = np.argsort(scores)[::-1][:10]
            scores = scores[rank_ID]
            results = [img_list[i].decode('utf-8') for i in rank_ID]
            end_time = time.time()
            score_type = 'Cosine Similarity'

        if request.form['algorithm'] == '2':
            nearest_neighbors = annoy_index.get_nns_by_vector(query_vector, 10, include_distances=True)
            results = [img_list[i].decode('utf-8') for i in nearest_neighbors[0]]
            scores = [item for item in nearest_neighbors[1]]
            end_time = time.time()
            score_type = 'Cosine Distance'

        return render_template('index.html',
                               image_paths_and_scores=zip(results, scores),
                               source=img_path,
                               query_time=f'{(end_time - start_time):.4f}',
                               page_status=1,
                               score_type=score_type)

    return render_template('index.html', page_status=2)


if __name__ == '__main__':
    app.run()
