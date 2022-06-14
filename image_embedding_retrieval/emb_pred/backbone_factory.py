import tensorflow as tf

class BackboneFactory:
    def __init__(self):
        self.map = {
            "resnet_50": tf.keras.applications.ResNet50,
            "vgg_19": tf.keras.applications.VGG19, 
            "efficientnet_b0":  tf.keras.applications.EfficientNetB0,
            "inception_v3": tf.keras.applications.InceptionV3,
            "densenet_121": tf.keras.applications.DenseNet121,
        }
        self.kernel_map = {
            "glorot_normal": tf.keras.initializers.GlorotNormal,
            "glorot_uniform": tf.keras.initializers.GlorotUniform,
            "he_normal": tf.keras.initializers.HeNormal,
            "he_uniform": tf.keras.initializers.HeUniform
        }
        self.kernel_init = tf.keras.initializers.GlorotNormal

    def __call__(self, key, input_tensor, weights):
        print("Use %s as weight init" % weights)
        backbone = self.map[key](include_top=False,
                                 weights=weights,
                                 input_tensor=input_tensor)

        return backbone
