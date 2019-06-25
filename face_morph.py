import argparse
import logging

import cv2
import numpy as np
from scipy import spatial
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
    parser.add_argument('--avg')
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
    boxes[:, 0] = boxes[:, 0] * frame.shape[1] + offset[0]
    boxes[:, 2] = boxes[:, 2] * frame.shape[1] + offset[0]
    boxes[:, 1] = boxes[:, 1] * frame.shape[0] + offset[1]
    boxes[:, 3] = boxes[:, 3] * frame.shape[0] + offset[1]
    return boxes


def crop_by_boxes(img, boxes):
    crops = []
    for box in boxes:
        cropped = crop_by_box(img, box)
        crops.append(cropped)
    return crops


def crop_by_box(img, box):
    ymin = int(max([box[1], 0]))
    ymax = int(min([box[3], img.shape[0]]))
    xmin = int(max([box[0], 0]))
    xmax = int(min([box[2], img.shape[1]]))
    return img[ymin:ymax, xmin:xmax]


def get_landmarks(model, face_img):
    input_name, input_shape = list(model.inputs.items())[0]
    output_name = list(model.outputs)[0]
    input_img = cv2.resize(face_img, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA)
    mean = []

    inference_frame = np.transpose(input_img, [2, 0, 1]).reshape(input_shape)
    outputs = model.predict({input_name: inference_frame})

    return outputs[output_name]


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, src_tri, dst_tri, size):
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(
        src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warp_triangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect


def face_morph(img, img2, tr_list1, tr_list2, alpha):
    img_morph = img.copy()

    for i in range(0, len(tr_list1), 1):
        # Get coordinates of a triangle
        t1 = [tr_list1[i][0], tr_list1[i][1], tr_list1[i][2]]
        t2 = [tr_list2[i][0], tr_list2[i][1], tr_list2[i][2]]
        # t = [trListM[i][0],trListM[i][1],trListM[i][2]]
        t = [tr_list1[i][0], tr_list1[i][1], tr_list1[i][2]]

        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        r = cv2.boundingRect(np.float32([t]))

        # Offset points by left top corner of the respective rectangles
        t1_rect = []
        t2_rect = []
        t_rect = []

        for i in range(0, 3):
            t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
            t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        # Get mask by filling triangle
        mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

        # Apply warpImage to small rectangular patches
        img1_rect = img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        size = (r[2], r[3])
        warp_image1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
        warp_image2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)

        # Alpha blend rectangular patches
        img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2

        # Copy triangular region of the rectangular patch to the output image
        img_morph[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = \
            img_morph[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + img_rect * mask
    return img_morph


if __name__ == '__main__':
    args = parse_args()

    input_img = cv2.imread(args.input)
    avg_img = cv2.imread(args.avg)

    drv = driver.load_driver('openvino')
    face_driver = drv()
    face_driver.load_model(args.face_model)

    landmarks_driver = drv()
    landmarks_driver.load_model(args.landmarks_model)

    face_boxes = get_boxes(face_driver, input_img)
    # avg_box = get_boxes(face_driver, avg_img, threshold=0.5)[0]
    # avg_face = crop_by_box(avg_img, avg_box)
    avg_face = avg_img
    face = crop_by_box(input_img, face_boxes[0])

    cv2.namedWindow("Image")
    cv2.imshow("Image", avg_face)
    cv2.waitKey(0)

    avg_landmarks = get_landmarks(landmarks_driver, avg_face).reshape(-1)
    face_landmarks = get_landmarks(landmarks_driver, face).reshape(-1)

    points = np.array(list(zip(avg_landmarks[::2], avg_landmarks[1::2])))
    face_points = np.array(list(zip(face_landmarks[::2], face_landmarks[1::2])))

    points[:, 0] = points[:, 0] * avg_face.shape[1]
    points[:, 1] = points[:, 1] * avg_face.shape[0]

    face_points[:, 0] = face_points[:, 0] * face.shape[1] + face_boxes[0][0]
    face_points[:, 1] = face_points[:, 1] * face.shape[0] + face_boxes[0][1]

    # for x, y in points:
    #     cv2.circle(avg_face, (int(x), int(y)), 3, (0, 0, 250), cv2.FILLED, cv2.LINE_AA)

    # cv2.imshow("Image", avg_face)
    # cv2.waitKey(0)

    input_img_circles = input_img.copy()
    for x, y in face_points:
        # x0 = int(x * input_img.shape[1])
        # y0 = int(y * input_img.shape[0])
        cv2.circle(input_img_circles, (int(x), int(y)), 3, (0, 0, 250), cv2.FILLED, cv2.LINE_AA)

    cv2.imshow("Image", input_img_circles)
    cv2.waitKey(0)

    # tri_avg = triangle.triangulate({'vertices': points})
    # tri_face = triangle.triangulate({'vertices': face_points})
    tri_avg = spatial.Delaunay(points)
    tri_face = spatial.Delaunay(face_points)

    tr_list1 = points[tri_avg.simplices]
    tr_list2 = face_points[tri_avg.simplices]
    # tr_list1 = points[tri_avg['triangles']]
    # tr_list2 = face_points[tri_avg['triangles']]

    # input_img_tri = input_img.copy()
    # for tr in tr_list2.astype(np.int):
    #     p1 = tuple(tr[0])
    #     p2 = tuple(tr[1])
    #     p3 = tuple(tr[2])
    #     cv2.line(input_img_tri, p1, p2, (250, 250, 250), 1, cv2.LINE_AA)
    #     cv2.line(input_img_tri, p2, p3, (250, 250, 250), 1, cv2.LINE_AA)
    #     cv2.line(input_img_tri, p3, p1, (250, 250, 250), 1, cv2.LINE_AA)
    #
    # cv2.imshow("Image", input_img_tri)
    # cv2.waitKey(0)

    morph = face_morph(input_img, avg_img, tr_list2, tr_list1, 0.5)

    cv2.imshow("Result", morph)
    cv2.waitKey(0)
