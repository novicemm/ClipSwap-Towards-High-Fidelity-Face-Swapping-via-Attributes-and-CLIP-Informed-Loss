import argparse
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization
from scripts.layers import *
from retinaface.models import *
from utils.utils import (estimate_norm, get_lm, inverse_estimate_norm,
                         norm_crop, transform_landmark_points)


def run_inference(opt, source, target, RetinaFace, ArcFace, ClipSwap, result_img_path, source_z=None):
    try:
        target = target if not isinstance(target, str) else cv2.imread(target)
        target = np.array(target)

        if source_z is None:
            source = cv2.imread(source)
            source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            source = np.array(source)

            source_h, source_w, _ = source.shape
            source_a = RetinaFace(np.expand_dims(source, axis=0)).numpy()[0]
            source_lm = get_lm(source_a, source_w, source_h)
            source_aligned = norm_crop(source, source_lm, image_size=112, shrink_factor=1.0)

            source_z = ArcFace.predict(np.expand_dims(source_aligned / 255.0, axis=0))


        blend_mask_base = np.zeros(shape=(256, 256, 1))
        blend_mask_base[77:240, 32:224] = 1
        blend_mask_base = gaussian_filter(blend_mask_base, sigma=7)

        im = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        im_h, im_w, _ = im.shape
        im_shape = (im_w, im_h)

        detection_scale = (im_w // 640) if (im_w > 640) else 1
        resized_im = cv2.resize(im, (im_w // detection_scale, im_h // detection_scale))
        faces = RetinaFace(np.expand_dims(resized_im, axis=0)).numpy()
        total_img = im / 255.0

        for annotation in faces:
            lm_align = get_lm(annotation, im_w, im_h)

            M, pose_index = estimate_norm(lm_align, 256, "arcface", shrink_factor=1.0)
            im_aligned = cv2.warpAffine(im, M, (256, 256), borderValue=0.0)

            face_swap = ClipSwap.predict([np.expand_dims((im_aligned - 127.5) / 127.5, axis=0), source_z])
            face_swap = (face_swap[0] + 1) / 2

            transformed_lmk = transform_landmark_points(M, lm_align)
            iM, _ = inverse_estimate_norm(lm_align, transformed_lmk, 256, "arcface", shrink_factor=1.0)
            iim_aligned = cv2.warpAffine(face_swap, iM, im_shape, borderValue=0.0)

            blend_mask = cv2.warpAffine(blend_mask_base, iM, im_shape, borderValue=0.0)
            blend_mask = np.expand_dims(blend_mask, axis=-1)

            total_img = (iim_aligned * blend_mask + total_img * (1 - blend_mask))

        total_img = np.clip(total_img * 255, 0, 255).astype('uint8')
        cv2.imwrite(result_img_path, cv2.cvtColor(total_img, cv2.COLOR_BGR2RGB))

        return total_img, source_z

    except Exception as e:
        print('\n', e)
        sys.exit(0)


class Configs():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--retina_path', type=str, default='./pretrained_ckpts/retinaface/RetinaFace-Res50.h5',
                                 help='Path to pretrained RetinaFace model file.')
        self.parser.add_argument('--arcface_path', type=str, default='./pretrained_ckpts/arcface/ArcFace-Res50.h5',
                                 help='Path to pretrained ArcFace model for identity extraction.')
        self.parser.add_argument('--model_path', type=str, default='./pretrained_ckpts/clipswap/clipswap.h5',
                                 help='Path to the pretrained ClipSwap model.')
        self.parser.add_argument('--target', type=str,
                                 help='Path to the target image file.')     
        self.parser.add_argument('--source', type=str,
                                 help='Path to the source image file.')   
        self.parser.add_argument('--output', type=str, default="output/result.jpg",
                                 help='Path to save the output image.')
       

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
    

if __name__ == '__main__':
    options = Configs().parse()

    output_folder = os.path.dirname(options.output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    available_gpus = tf.config.list_physical_devices('GPU')
    if available_gpus:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpu_devices[0], 'GPU')


    print('\nStart!!!')
 

    retinaface_model = load_model(
        options.retina_path,
        compile=False,
        custom_objects={
            "FPN": FPN,
            "SSH": SSH,
            "BboxHead": BboxHead,
            "LandmarkHead": LandmarkHead,
            "ClassHead": ClassHead
        }
    )

    arcface_model = load_model(options.arcface_path, compile=False)

    generator_model = load_model(
        options.model_path,
        compile=False,
        custom_objects={
            "AdaIN": AdaIN,
            "AdaptiveAttention": AdaptiveAttention,
            "InstanceNormalization": InstanceNormalization
        }
    )


    print(f'\nProcessing target: {options.target}')
    
    run_inference(
        options,
        options.source,
        options.target,
        retinaface_model,
        arcface_model,
        generator_model,
        options.output
    )

    print(f'\nOutput saved at: {options.output}')
