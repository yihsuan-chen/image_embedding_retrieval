import os,sys
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from image_embedding_retrieval.emb_pred.emb_predictor import EmbPredictor
from annoy import AnnoyIndex
from box import Box
import numpy as np
import cv2
import argparse
import json
import time



def retrieve(config, img_path, k):
    emb_predictor = EmbPredictor(config.emb.input_shape, config.emb.backbone, config.emb.weight_init)
    u = AnnoyIndex(emb_predictor.out_dim, 'angular')
    u.load(os.path.join(config.ann.idx_save_path, 'index.ann'))
    # print('u.get_n_items()')
    # print(u.get_n_items())
    # print('u.get_n_trees()')
    # print(u.get_n_trees())
    #img_path = os.path.join(config.img_root, img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, tuple(config.emb.input_shape[:2]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = np.expand_dims(img, axis=0)
    query_emb = emb_predictor.pred(img)
    query_emb = query_emb.numpy()[0]
    values = u.get_nns_by_vector(query_emb,
                                 k,
                                 search_k=-1,
                                 include_distances=True)
    idxs, dists = values
    img_names = idxs_to_names(
        os.path.join(config.emb_save_path, 'img_names.json'), idxs)
    dists = np.asarray(dists)
    return img_names, dists


def idxs_to_names(img_names_path, idxs):
    with open(img_names_path) as f:
        img_names = json.loads(f.read())
    results = []
    for idx in idxs:
        results.append(img_names[idx])
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='retrieval settings')
    parser.add_argument('--config', help='config path', default='config.json')
    parser.add_argument('--k',
                        help='top k most similar images',
                        default=10,
                        type=int)
    parser.add_argument('--img_path', help='query img path', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = Box(json.loads(f.read()))
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    img_names, dists = retrieve(config, args.img_path, args.k)