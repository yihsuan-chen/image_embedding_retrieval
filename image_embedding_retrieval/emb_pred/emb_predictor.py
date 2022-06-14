
import tensorflow_addons as tfa
import tensorflow as tf
from image_embedding_retrieval.emb_pred.backbone_factory import BackboneFactory

optimizer = tfa.optimizers.SGDW


class EmbPredictor():
    def __init__(self, input_shape,backbone_name, weight_init):
        self.input_shape = input_shape
        self.model, self.out_dim = self._get_model(input_shape,backbone_name, weight_init)

    @tf.function
    def pred(self, imgs):
        embs = self.model(imgs, training=False)
        return embs

    def _get_model(self, input_shape,backbone_name, weight_init='imagenet'):
        backbone_factory = BackboneFactory()
        input_tensor = tf.keras.Input(shape=input_shape, batch_size=None)
        backbone = backbone_factory(backbone_name, input_tensor, weight_init)
        fmap = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(backbone.output)        
        out_dim = fmap.get_shape().as_list()[-1]
        model = tf.keras.Model(backbone.input, fmap)
        return model, out_dim