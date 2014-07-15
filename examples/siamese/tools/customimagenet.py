import caffe
import caffe.imagenet
from skimage import io
from skimage import transform
import numpy as np
import os
   
IMAGE_DIM = 256
CROPPED_DIM = 227

# Load the imagenet mean file
IMAGENET_MEAN = np.load(
    os.path.join(os.path.dirname(__file__)+'/../distribute/python/caffe/imagenet', 'ilsvrc_2012_mean.npy'))

class CustomImageNetClassifier(caffe.imagenet.ImageNetClassifier):
    def __init__(self, model_def_file, pretrained_model, center_only=False, num_output=1000):
        super(CustomImageNetClassifier, self).__init__(model_def_file, pretrained_model, center_only, num_output)

    def predict_from_depth(self, image):
        """Input is a skimage or any numpy array, RGB format"""
        img = io.imread(image, as_grey=True)
        if img.ndim == 2:
          img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        elif img.shape[2] == 4:
          img = img[:, :, :3]
        # Resize and convert to BGR
        img_reshape = (transform.resize(img, (IMAGE_DIM,IMAGE_DIM)) * 255)[:, :, ::-1]
        # subtract main
        img_reshape -= IMAGENET_MEAN
        input_blob = [caffe.imagenet.oversample(img_reshape, self._center_only)]
        self.caffenet.Forward(input_blob, self._output_blobs)
        return self._output_blobs[0].mean(0).flatten()
        

