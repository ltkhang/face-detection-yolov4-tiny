import cv2
import time
import darknet


def draw_boxes(detections, image, source_shape):
    s_w, s_h = source_shape # source, eg 416x416
    t_h, t_w, _ = image.shape # target
    w_scale = float(t_w) / s_w
    h_scale = float(t_h) / s_h

    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        x = x * w_scale
        y = y * h_scale
        w = w * w_scale
        h = h * h_scale
        left, top, right, bottom = darknet.bbox2points((x,y,w,h))
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
    return image


if __name__ == '__main__':
    network, class_names, colors = darknet.load_network("yolo/yolov4-tiny-3l.cfg", "yolo/obj.data",
                                                "yolo/yolov4-tiny-3l_best.weights")
    vid_path = 'vid-apec.mp4'
    cap = cv2.VideoCapture(vid_path)
    vid_width = int(cap.get(3))
    vid_height = int(cap.get(4))
    vid_fps = cap.get(5)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    # setting write video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, vid_fps, (vid_width, vid_height))
    while True:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height),
                                       interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            t1 = time.time()
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5)
            dt = time.time() - t1
            frame = cv2.putText(frame, 'Detection time: {:.4f}, fps: {}'.format(dt, 1 / dt), (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2)
            frame = draw_boxes(detections, frame, (width, height))
            cv2.imshow('Detect', frame)
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
