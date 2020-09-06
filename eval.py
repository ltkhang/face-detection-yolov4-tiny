import cv2
import time
import darknet
from glob import glob
import os


def rescale(detections, image, source_shape):
    s_w, s_h = source_shape # source, eg 416x416
    t_h, t_w, _ = image.shape # target
    w_scale = float(t_w) / s_w
    h_scale = float(t_h) / s_h
    res = []
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        x = x * w_scale
        y = y * h_scale
        w = w * w_scale
        h = h * h_scale
        left, top, right, bottom = darknet.bbox2points((x,y,w,h))
        res.append((left, top, right - left, bottom - top, confidence))
    return res


if __name__ == '__main__':
    network, class_names, colors = darknet.load_network("yolo/yolov4-tiny-3l.cfg", "yolo/obj.data",
                                                "yolo/yolov4-tiny-3l_best.weights")
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    #
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_resized = cv2.resize(frame_rgb, (width, height),
    #                            interpolation=cv2.INTER_LINEAR)
    # darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    # detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.3)
    # frame = draw_boxes(detections, frame, (width, height))
    val_dir = '/home/khang/Khang/data/wider/val/WIDER_val/images'
    dirs = []
    for dir_name in os.listdir(val_dir):
        if os.path.isdir(os.path.join(val_dir, dir_name)):
            # os.mkdir(os.path.join('eval', dir_name))
            for img_file in glob(os.path.join(val_dir, dir_name, '*.jpg')):
                f = open(os.path.join('eval', dir_name, os.path.basename(img_file)[:-4] + '.txt'), 'w')
                frame = cv2.imread(img_file)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
                darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
                # h, w, _ = frame.shape
                # darknet_image = darknet.make_image(w, h, 3)
                # darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
                detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.3)
                # res = rescale(detections, frame, (w,h))
                res = rescale(detections, frame, (width, height))
                f.write('{}\n'.format(os.path.basename(img_file)[:-4]))
                f.write('{}\n'.format(len(res)))
                for r in res:
                    l, t, w, h, s = r
                    f.write('{} {} {} {} {:.4f}\n'.format(l, t, w, h, float(s) / 100.0))
                f.close()

