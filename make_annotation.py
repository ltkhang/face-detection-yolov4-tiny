import os
from glob import glob
import cv2


def strip_str(s):
    return s.rstrip('\n').strip()


def gen_txt(img_dir, gt_file, outfile):
    f = open(gt_file, 'r')
    lines = f.readlines()
    f.close()
    i = 0
    fo = open(outfile, 'w')
    while i < len(lines):
        img_path = strip_str(lines[i])
        fo.write('{}\n'.format(os.path.join(img_dir, img_path)))
        img = cv2.imread(os.path.join(img_dir, img_path))
        img_h, img_w, _ = img.shape
        f = open(os.path.join(img_dir, img_path[:-4] + '.txt'), 'w')
        i += 1
        num_box = int(strip_str(lines[i]))
        i +=1
        for j in range(num_box):
            anno_str = strip_str(lines[i + j])
            x, y, w, h, blur, expression, illumination, invalid, occlusion, pose = anno_str.split(' ')
            invalid = int(invalid)
            if not invalid:
                x = float(x); y = float(y); w = float(w); h = float(h)
                x_center = (x + w/2.0) / img_w
                y_center = (y + h/2.0) / img_h
                w_norm = w / float(img_w)
                h_norm = h / float(img_h)
                f.write('0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(x_center, y_center, w_norm, h_norm))
        #         img = cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
        # cv2.imshow('abc', img)
        # if cv2.waitKey(0) & 0xFF == 'q':
        #     break

        i += num_box
        if num_box == 0:
            i += 1
        f.close()
    fo.close()


if __name__ == '__main__':

    train_set_dir = '/home/khang/Khang/data/wider/train/WIDER_train/images'
    val_set_dir = '/home/khang/Khang/data/wider/val/WIDER_val/images'

    train_gt = '/home/khang/Khang/data/wider/wider_face_split/wider_face_train_bbx_gt.txt'
    val_gt = '/home/khang/Khang/data/wider/wider_face_split/wider_face_val_bbx_gt.txt'

    gen_txt(train_set_dir, train_gt, 'train.txt')
    gen_txt(val_set_dir, val_gt, 'val.txt')





