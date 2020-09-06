from ctypes import *
import random
import cv2
# from sort import *
import numpy as np
import sys
import time


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    if isinstance(image, bytes):
        # image is a filename
        # i.e. image = b'/darknet/data/dog.jpg'
        im = load_image(image, 0, 0)
    else:
        # image is an nparray
        # i.e. image = cv2.imread('/darknet/data/dog.jpg')
        im, image = array_to_image(image)
        rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                            (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    return res


def convert_back(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


# if __name__ == "__main__":
    # mot_tracker = Sort()
    # net = load_net(b"cfg/yolov2.cfg", b"yolov2.weights", 0)
    # meta = load_meta(b"cfg/coco.data")
    #
    # rtsp = 'rtsp://192.168.10.101:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
    # if (len(sys.argv) > 1):
    #     rtsp = sys.argv[1]
    # cap = cv2.VideoCapture(rtsp)
    # width = int(cap.get(3))
    # height = int(cap.get(4))
    # fps = cap.get(5)
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    # while True:
    #     ret, frame = cap.read()
    #     frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    #     r = detect(net, meta, frame)
    #     detections = np.array([np.zeros(5)])
    #     for item in r:
    #         if (item[0].decode() != 'person'):
    #             continue
    #         x = int(item[2][0])
    #         y = int(item[2][1])
    #         w = int(item[2][2])
    #         h = int(item[2][3])
    #         xmin, ymin, xmax, ymax = convert_back(float(x), float(y), float(w), float(h))
    #         score = item[1]
    #         detections = np.append(detections, [[xmin, ymin, xmax, ymax, score]], axis=0)
    #     print(detections)
    #     track_bbs_ids = mot_tracker.update(detections)
    #     for item in track_bbs_ids:
    #         item = item.astype(np.int32)
    #         pt1 = (item[0], item[1])
    #         pt2 = (item[2], item[3])
    #         cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
    #         cv2.putText(frame, "ID: [" + str(int(round(item[4]))) + "]", (pt1[0], pt1[1] + 20),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 3)
    #     cv2.imshow("Detect", frame)
    #     out.write(frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    net = load_net(b"yolo/yolov4-tiny.cfg", b"../darknet/yolov4-tiny.weights", 0)
    meta = load_meta(b"yolo/coco.data")
    vid_path = 'vid.mp4'
    cap = cv2.VideoCapture(vid_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)
    while True:
        t1 = time.time()
        ret, frame = cap.read()
        if ret:
            r = detect(net, meta, frame)
            for item in r:
                x = int(item[2][0])
                y = int(item[2][1])
                w = int(item[2][2])
                h = int(item[2][3])
                xmin, ymin, xmax, ymax = convert_back(float(x), float(y), float(w), float(h))
                score = item[1]
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            delta_t = time.time() - t1
            frame = cv2.putText(frame, 'Detect time: {:.4f}, fps: {}'.format(delta_t, 1/delta_t), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 3)
            cv2.imshow('Detect', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


