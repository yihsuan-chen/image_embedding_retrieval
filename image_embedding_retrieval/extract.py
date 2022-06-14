from annoy import AnnoyIndex
from box import Box
from tqdm import tqdm
import numpy as np
import argparse
import json
import os,sys
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import tensorflow as tf
from image_embedding_retrieval.emb_pred.emb_predictor import EmbPredictor



class Emb_Generator():
  
  def __init__(self,  backbone_name='densenet_121',input_size: list=[224,224,3],weight_init ='imagenet',gpu: int=-1):
    self.gpu = str(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
    self.input_size = input_size
    self.backbone_name = backbone_name
    self.weight_init = weight_init
    self.emb_model = EmbPredictor(self.input_size,self.backbone_name, self.weight_init)

    
    """
    Args:
      backbone_name(str): model_path of embedding pretext weight
      input_size(list): model image input size 
      weight_init(str): imagenet init or others
      gpu(int): visible gpu number (-1: for cpu only)
    """     
    
  # read and get emb from list of image-path
  def get_total_emb(self, img_paths, batch_size =32):
    def read_img(img_paths):
      def read(img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(contents=img, channels=3)
        img = tf.image.resize(img, tuple(self.input_size[:-1]))
        return img
      imgs = tf.map_fn(read, img_paths, fn_output_signature=tf.float32)
      return imgs

    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.batch(batch_size, drop_remainder=False).map(
      lambda x: read_img(x), num_parallel_calls=tf.data.AUTOTUNE)
    total_embs = []
    for elem in tqdm(dataset):
        embs = self.emb_model.pred(elem)
        total_embs.append(embs)
    total_embs = np.concatenate(total_embs, axis=0)
    return total_embs

def gen_ann(config):
    embs = np.load(os.path.join(config.emb_save_path, 'embeds.npy'))
    dim = embs.shape[-1]
    t = AnnoyIndex(dim, 'angular')
    for idx, emb in enumerate(embs):
        t.add_item(idx, emb)
    t.build(config.ann.tree_num)
    ann_save_path = os.path.join(config.ann.idx_save_path)
    if not os.path.exists(ann_save_path):
        os.makedirs(ann_save_path)
    t.save(os.path.join(ann_save_path, 'index.ann'))
    print('ANN related data save in %s' % config.ann.idx_save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='retrieval settings')
    parser.add_argument('--config', help='config path', default='config.json')
    parser.add_argument('--emb', action='store_true')
    parser.add_argument('--ann', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  with open(args.config) as f:
      config = Box(json.loads(f.read()))
  img_paths = os.listdir(config.img_root)
  img_paths = [os.path.join(config.img_root, file) for file in img_paths if file.endswith('.jpg')]

  if args.emb:
      extract_emb = Emb_Generator(config.emb.backbone, config.emb.input_shape, config.emb.weight_init)
      output = extract_emb.get_total_emb(img_paths)
      if not os.path.exists(config.emb_save_path):
            os.makedirs(config.emb_save_path)
      np.save(  os.path.join(config.emb_save_path,'embeds.npy'),output)
      with open(os.path.join(config.emb_save_path, 'img_names.json'), 'w') as f:
        json.dump(img_paths , f)
  if args.ann:
    gen_ann(config)