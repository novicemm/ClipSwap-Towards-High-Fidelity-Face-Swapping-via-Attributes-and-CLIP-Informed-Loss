import tensorflow as tf
from tensorflow.keras import layers, Model
import math
import numpy as np

class ResBlk(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_out, actv=None, normalize=False, downsample=False):
        super(ResBlk, self).__init__()
        if actv is None:
            self.actv = layers.LeakyReLU(0.2)
        else:
            self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out

        self.conv1 = layers.Conv2D(dim_in, kernel_size=3, strides=1, padding='same')
        self.conv2 = layers.Conv2D(dim_out, kernel_size=3, strides=1, padding='same')
        
        if self.normalize:
            self.norm1 = layers.LayerNormalization(axis=-1)
            self.norm2 = layers.LayerNormalization(axis=-1)
            
        if self.learned_sc:
            self.conv1x1 = layers.Conv2D(dim_out, kernel_size=1, strides=1, padding='valid', use_bias=False)
    
    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = tf.nn.avg_pool(x, ksize=2, strides=2, padding='SAME')
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = tf.nn.avg_pool(x, ksize=2, strides=2, padding='SAME')
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def call(self, x):
        return (self._shortcut(x) + self._residual(x)) / math.sqrt(2)


class AttrEncoder(tf.keras.Model):
    def __init__(self, img_size=256, max_conv_dim=512):
        super(AttrEncoder, self).__init__()
        dim_in = 2**14 // img_size
        blocks = [layers.Conv2D(dim_in, kernel_size=3, strides=1, padding='same')]
        repeat_num = int(np.log2(img_size)) - 4
        repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            dim_in = dim_out
        for _ in range(2):
            blocks.append(ResBlk(dim_out, dim_out, normalize=True))
        self.model = tf.keras.Sequential(blocks)

    def call(self, x, masks=None):
        attr_all_features = []
        cache = {}
        for block in self.model.layers:
            if (masks is not None) and (x.shape[1] in [32, 64, 128]):
                cache[x.shape[1]] = x
            x = block(x)
            attr_all_features.append(x)
        return x, attr_all_features, cache