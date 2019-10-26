__all__ = ['Model']


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
# import tkinter

od_path = os.path.abspath('./tfmodel')
sys.path.append(od_path) # Object Detection API
print(sys.path)

# from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class Model:
    def __init__(self, model_name: str = 'ssd_mobilenet_v1_coco_2017_11_17',
                 download_base: str = 'http://download.tensorflow.org/models/object_detection/',
                 num_classes: int = 90):
        # What model to download.
        self._model_name = model_name
        # Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
        self._download_base = download_base
        # Number of classes to detect
        self._num_classes = num_classes

        self._download_model_if_needed()
        self._setup_labels()
        self._init_session()
        self._prepare_tensor_dict()


    @property
    def model_file(self):
        return self._model_name + '.tar.gz'

    @property
    def ckpt_file(self):
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        return self._model_name + '/frozen_inference_graph.pb'

    @property
    def labels_file(self):
        # List of the strings that is used to add correct label for each box.
        cur_dir = os.path.dirname(__file__)
        return os.path.join(cur_dir, 'bbox-label-map', 'mscoco_label_map.pbtxt')

    def _download_model_if_needed(self):
        # Download Model
        opener = urllib.request.URLopener()
        if not os.path.isfile(self.model_file):
            opener.retrieve(self._download_base + self.model_file, self.model_file)
        if not os.path.isfile(self.ckpt_file):
            tar_file = tarfile.open(self.model_file)
            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
                if 'frozen_inference_graph.pb' in file_name:
                    tar_file.extract(file, os.getcwd())

    def _init_session(self):
        # Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.ckpt_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.sess = tf.Session(graph=self.detection_graph)

    def __del__(self):
        self.sess.close()
        print('TensorFlow session closed.')

    def _prepare_tensor_dict(self):
        # Get handles to input and output tensors
        # ops = tf.get_default_graph().get_operations()
        ops = self.sess.graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                # tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                tensor_dict[key] = self.sess.graph.get_tensor_by_name(tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image 
            # coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[1], image.shape[2])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        # image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        image_tensor = self.sess.graph.get_tensor_by_name('image_tensor:0')

        self._tensor_dict = tensor_dict
        self._image_tensor = image_tensor

    def _setup_labels(self):
        # Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, 
        # we know that this corresponds to `airplane`.  Here we use internal utility functions, 
        # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(self.labels_file)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=self._num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def infer(self, image: np.ndarray) -> dict:
        """ Model expects image to have shape: [1, None, None, 3] """
        # Run inference
        output_dict = self.sess.run(self._tensor_dict,
                            feed_dict={self._image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.int64)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        self.output_dict = output_dict
        return output_dict

    def visualize_boxes_and_labels_on_image(self, image_np, output_dict=None):
        # Visualization of the results of a detection.
        if not output_dict:
            output_dict = self.output_dict
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        return image_np
