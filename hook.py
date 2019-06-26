import json
import logging

import cv2
from ml_serving.utils import helpers

import beautify


LOG = logging.getLogger(__name__)
PARAMS = {
    'shape-model-path': '',
    'threshold': 0.5,
}
landmarks_driver = None


def init_hook(**params):
    PARAMS.update(params)
    PARAMS['threshold'] = float(PARAMS['threshold'])

    LOG.info('Initialized with params: ')
    LOG.info(json.dumps(PARAMS, indent=2))

    LOG.info("Loading landmarks driver...")
    global landmarks_driver
    landmarks_driver = beautify.load_shape_model(PARAMS['shape-model-path'])
    LOG.info("Loaded.")


def process(inputs, ctx, **kwargs):
    face_driver = ctx.driver

    image, is_video = helpers.load_image(inputs, 'input', rgb=False)
    boxes = beautify.get_boxes(face_driver, image, PARAMS['threshold'])

    for box in boxes:
        image = beautify.beauty(landmarks_driver, image, box)

    if is_video:
        return {'output': image}
    else:
        image_bytes = cv2.imencode('.jpg', image)[1].tostring()
        return {'output': image_bytes}
