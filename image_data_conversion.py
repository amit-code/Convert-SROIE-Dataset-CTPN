import json
import random
import numpy as np
import os
import shutil
import sys
import cv2 as cv
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from shapely.geometry import Polygon


os.chdir("/home/amit/text-detection-ctpn-banjin-dev/data/dataset/")


# os.mkdir("/home/amit/text-detection-ctpn-banjin-dev/data/dataset/mlt_resized")
# os.mkdir("/home/amit/text-detection-ctpn-banjin-dev/data/dataset/mlt_resized/image")
# os.mkdir("/home/amit/text-detection-ctpn-banjin-dev/data/dataset/mlt_resized/label")


def pickTopLeft(poly):
    idx = np.argsort(poly[:, 0])
    if poly[idx[0], 1] < poly[idx[1], 1]:
        s = idx[0]
    else:
        s = idx[1]

    return poly[(s, (s + 1) % 4, (s + 2) % 4, (s + 3) % 4), :]


def orderConvex(p):
    points = Polygon(p).convex_hull
    points = np.array(points.exterior.coords)[:4]
    points = points[::-1]
    points = pickTopLeft(points)
    points = np.array(points).reshape([4, 2])
    return points


def shrink_poly(poly, r=16):
    # y = kx + b
    x_min = int(np.min(poly[:, 0]))
    x_max = int(np.max(poly[:, 0]))

    k1 = (poly[1][1] - poly[0][1]) / (poly[1][0] - poly[0][0])
    b1 = poly[0][1] - k1 * poly[0][0]

    k2 = (poly[2][1] - poly[3][1]) / (poly[2][0] - poly[3][0])
    b2 = poly[3][1] - k2 * poly[3][0]

    res = []

    start = int((x_min // 16 + 1) * 16)
    end = int((x_max // 16) * 16)

    p = x_min
    res.append([p, int(k1 * p + b1),
                start - 1, int(k1 * (p + 15) + b1),
                start - 1, int(k2 * (p + 15) + b2),
                p, int(k2 * p + b2)])

    for p in range(start, end + 1, r):
        res.append([p, int(k1 * p + b1),
                    (p + 15), int(k1 * (p + 15) + b1),
                    (p + 15), int(k2 * (p + 15) + b2),
                    p, int(k2 * p + b2)])
    return np.array(res, dtype=np.int).reshape([-1, 8])


DATA_FOLDER = "/home/amit/text-detection-ctpn-banjin-dev/data/dataset/image-data/image-data-new/"
OUTPUT = "/home/amit/text-detection-ctpn-banjin-dev/data/dataset/mlt_resized/"
MAX_LEN = 1200
MIN_LEN = 600

im_fns = os.listdir(os.path.join(DATA_FOLDER, "image"))
im_fns.sort()


for im_fn in tqdm(im_fns):
    try:
        _, fn = os.path.split(im_fn)
        bfn, ext = os.path.splitext(fn)
        if ext.lower() not in ['.jpg', '.png']:
            continue
        gt_path = DATA_FOLDER+"label"+"/"+bfn+".txt"
        # gt_path = os.path.join(DATA_FOLDER, "label", 'gt_' + bfn + '.txt')
        img_path = os.path.join(DATA_FOLDER, "image", im_fn)

        img = cv.imread(img_path)
        img_size = img.shape
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])

        im_scale = float(600) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 1200:
            im_scale = float(1200) / float(im_size_max)
        new_h = int(img_size[0] * im_scale)
        new_w = int(img_size[1] * im_scale)

        new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
        new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

        re_im = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        re_size = re_im.shape

        polys = []
        with open(gt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            splitted_line = line.strip().lower().split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, splitted_line[:8])
            poly = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])
            poly[:, 0] = poly[:, 0] / img_size[1] * re_size[1]
            poly[:, 1] = poly[:, 1] / img_size[0] * re_size[0]
            poly = orderConvex(poly)
            polys.append(poly)

            # cv.polylines(re_im, [poly.astype(np.int32).reshape((-1, 1, 2))], True,color=(0, 255, 0), thickness=2)

        res_polys = []
        for poly in polys:
            # delete polys with width less than 10 pixel
            if np.linalg.norm(poly[0] - poly[1]) < 10 or np.linalg.norm(poly[3] - poly[0]) < 10:
                continue

            res = shrink_poly(poly)
            # for p in res:
            #    cv.polylines(re_im, [p.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=1)

            res = res.reshape([-1, 4, 2])
            for r in res:
                x_min = np.min(r[:, 0])
                y_min = np.min(r[:, 1])
                x_max = np.max(r[:, 0])
                y_max = np.max(r[:, 1])

                res_polys.append([x_min, y_min, x_max, y_max])

        cv.imwrite(os.path.join(OUTPUT, "image", fn), re_im)
        write_text_path = OUTPUT+"label"+"/"+bfn+".txt"
        # with open(os.path.join(OUTPUT, "label", bfn) + ".txt", "w") as f:
        with open(write_text_path, "w") as f:
            for p in res_polys:
                line = ",".join(str(p[i]) for i in range(4))
                f.writelines(line + "\r\n")
                # for p in res_polys:
                #    cv.rectangle(re_im,(p[0],p[1]),(p[2],p[3]),color=(0,0,255),thickness=1)

                # cv.imshow("demo",re_im)
                # cv.waitKey(0)
    except Exception as e:
      print(e)
      print("Error processing {}".format(im_fn))
