import argparse
import logging

import cv2
import dlib
import numpy as np
from scipy import spatial
from scipy import interpolate
from ml_serving.drivers import driver

logging.basicConfig(
    format='%(asctime)s %(levelname)-5s %(name)-10s [-] %(message)s',
    level='INFO'
)
LOG = logging.getLogger(__name__)
threshold = 0.9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face-model')
    parser.add_argument('--landmarks-model')
    parser.add_argument('--input')
    parser.add_argument('--output')

    return parser.parse_args()


def get_boxes(face_driver, frame, threshold=0.5, offset=(0, 0)):
    input_name, input_shape = list(face_driver.inputs.items())[0]
    output_name = list(face_driver.outputs)[0]
    inference_frame = cv2.resize(frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA)
    inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
    outputs = face_driver.predict({input_name: inference_frame})
    output = outputs[output_name]
    output = output.reshape(-1, 7)
    bboxes_raw = output[output[:, 2] > threshold]
    # Extract 5 values
    boxes = bboxes_raw[:, 3:7]
    confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
    boxes = np.concatenate((boxes, confidence), axis=1)
    # Assign confidence to 4th
    # boxes[:, 4] = bboxes_raw[:, 2]
    xmin = boxes[:, 0] * frame.shape[1] + offset[0]
    xmax = boxes[:, 2] * frame.shape[1] + offset[0]
    ymin = boxes[:, 1] * frame.shape[0] + offset[1]
    ymax = boxes[:, 3] * frame.shape[0] + offset[1]
    xmin[xmin < 0] = 0
    xmax[xmax > frame.shape[1]] = frame.shape[1]
    ymin[ymin < 0] = 0
    ymax[ymax > frame.shape[0]] = frame.shape[0]

    boxes[:, 0] = xmin
    boxes[:, 2] = xmax
    boxes[:, 1] = ymin
    boxes[:, 3] = ymax
    return boxes


def crop_by_boxes(img, boxes):
    crops = []
    for box in boxes:
        cropped = crop_by_box(img, box)
        crops.append(cropped)
    return crops


def crop_by_box(img, box, margin=0):
    h = (box[3] - box[1])
    w = (box[2] - box[0])
    ymin = int(max([box[1] - h * margin, 0]))
    ymax = int(min([box[3] + h * margin, img.shape[0]]))
    xmin = int(max([box[0] - w * margin, 0]))
    xmax = int(min([box[2] + w * margin, img.shape[1]]))
    return img[ymin:ymax, xmin:xmax]


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def get_landmarks(model, frame, box):
    box = box.astype(np.int)
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    shape = model(frame, rect)
    shape = shape_to_np(shape)

    return shape


def draw_points(img, points, color=(0, 0, 250)):
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), 3, color, cv2.FILLED, cv2.LINE_AA)


def select_contour(contour, mask, img, box, color=0, draw=False):
    tck, u = interpolate.splprep(contour.transpose(), u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 100)
    xs, ys = interpolate.splev(u_new, tck, der=0)
    points = np.stack([xs, ys]).transpose()
    relative_points = points.copy()
    relative_points[:, 0] -= box[0]
    relative_points[:, 1] -= box[1]

    if draw:
        cv2.fillConvexPoly(img, points.astype(int), (250, 250, 250), cv2.LINE_AA)

    cv2.fillConvexPoly(mask, np.round(relative_points).astype(int), color, cv2.LINE_AA)
    cv2.polylines(
        mask,
        [np.round(relative_points).astype(np.int32)],
        1, color, thickness=3, lineType=cv2.LINE_AA
    )


def beauty(landmarks_driver, img, face_box):
    face_landmarks = get_landmarks(landmarks_driver, img, face_box)

    left_eye = face_landmarks[36:42]
    right_eye = face_landmarks[42:48]
    mouth = face_landmarks[48:60]
    left_brow = face_landmarks[17:22]
    right_brow = face_landmarks[22:27]
    # left_chin = face_landmarks[0:5] + face_landmarks[31:32] + face_landmarks[36:37]

    # draw_points(img, left_eye, (250, 0, 0))
    # draw_points(img, right_eye, (0, 250, 0))
    # draw_points(img, mouth, (250, 0, 250))
    # draw_points(img, left_brow, (250, 250, 0))
    # draw_points(img, right_brow, (0, 250, 250))
    # draw_points(img, left_chin, (0, 250, 250))

    face = crop_by_box(img, face_box)
    mask = np.ones_like(face[:, :, 0])
    mask = np.expand_dims(mask, axis=2)

    contours = [left_eye, right_eye, mouth, left_brow, right_brow]
    for c in contours:
        try:
            select_contour(c, mask, img, face_box, draw=False)
        except Exception as e:
            LOG.warning("Error during select contour(interpolate): %s" % e)

    # select_contour(left_chin, mask, img, face_box, draw=True)
    # res = cv2.bitwise_and(face, face, mask=mask.squeeze())
    # cv2.imshow("Res", res)
    # cv2.waitKey(0)
    # draw_points(img, points, (255, 255, 255))

    face_y = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)

    for i in range(2):
        y_vals = face_y[:, :, 0:1][mask > 0]
        cr_vals = face_y[:, :, 1:2][mask > 0]
        cb_vals = face_y[:, :, 2:3][mask > 0]

        y_mean, y_std = y_vals.mean(), y_vals.std()
        cr_mean, cr_std = cr_vals.mean(), cr_vals.std()
        cb_mean, cb_std = cb_vals.mean(), cb_vals.std()

        factor = [2., 2.2, 2.2]
        indices_y = np.where(
            (y_mean - y_std * factor[0] > face_y[:, :, 0:1]) | (face_y[:, :, 0:1] > y_mean + y_std * factor[0])
        )[:2]
        indices_cr = np.where(
            (cr_mean - cr_std * factor[1] > face_y[:, :, 1:2]) | (face_y[:, :, 1:2] > cr_mean + cr_std * factor[1])
        )[:2]
        indices_cb = np.where(
            (cb_mean - cb_std * factor[2] > face_y[:, :, 2:3]) | (face_y[:, :, 2:3] > cb_mean + cb_std * factor[2])
        )[:2]

        # Remove pixels from mask
        mask[indices_y] = 0
        mask[indices_cb] = 0
        mask[indices_cr] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # res = cv2.bitwise_and(face, face, mask=mask)
    # cv2.imshow("Res", res)
    # cv2.waitKey(0)

    size = face.shape[1] // 27
    size_blur = face.shape[1] // 20
    size_odd = size_blur if size_blur % 2 != 0 else size_blur - 1
    mask = mask * 255
    mask = cv2.GaussianBlur(mask, (size_odd, size_odd), 0)
    # __import__('ipdb').set_trace()
    mask = np.expand_dims(mask, axis=2)
    # tone = cv2.GaussianBlur(face_y, (15, 15), 0)

    # texture = np.clip(tone[:, :, 0:1] - face_y[:, :, 0:1] * 0.5, 0, 255).astype(np.uint8)

    # Multiply the background with ( alpha )
    # res = (mask.astype(float) / 255 * face).astype(np.uint8)
    # cv2.imshow("Res", res)
    # cv2.waitKey(0)

    filtered = cv2.bilateralFilter(face, size if size >= 9 else 9, 75, 75)
    mixed = cv2.seamlessClone(
        filtered, face, mask, (face.shape[1] // 2, face.shape[0] // 2), cv2.NORMAL_CLONE
    )

    # cv2.imshow("Res", mixed)
    # cv2.waitKey(0)

    center_box = (int(face_box[0] + face.shape[1] / 2), int(face_box[1] + face.shape[0] / 2))

    mixed_mask = np.ones_like(mixed) * 255
    result = cv2.seamlessClone(mixed, img, mixed_mask, center_box, cv2.NORMAL_CLONE)
    # cv2.waitKey(0)
    # cv2.imshow("Result", result)
    # cv2.waitKey(0)
    return result


def load_models(face_path, shape_path):
    drv = driver.load_driver('openvino')
    face_driver = drv()
    face_driver.load_model(face_path)

    landmarks_driver = load_shape_model(shape_path)

    return face_driver, landmarks_driver


def load_shape_model(shape_path):
    return dlib.shape_predictor(shape_path)


def main():
    args = parse_args()

    input_img = cv2.imread(args.input)

    face_driver, landmarks_driver = load_models(args.face_model, args.landmarks_model)

    print('Models loaded.')

    face_boxes = get_boxes(face_driver, input_img)

    filtered = input_img.copy()
    for box in face_boxes:
        filtered = beauty(landmarks_driver, filtered, box)

    cv2.imshow("Image", input_img)
    cv2.imshow("Result", filtered)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

