#!/usr/bin/env python3
# coding: utf-8
import runway
from runway.data_types import number, text, image
import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn
from PIL import Image
import io


STD_SIZE = 120

mode = 'gpu'
dlib_landmark = False
dlib_bbox = True #changed
dump_obj = True
paf_size = 3
dump_paf = False
dump_pncc = True
dump_depth = True
dump_pose = True
dump_roi_box = False
dump_pts = True
dump_ply = True
dump_vertex = False
dump_res = True
bbox_init = 'one'
show_flg = True

"""
add run options for face points, pose estimation (reconstruction?)
"""


@runway.setup(options={'checkpoint': runway.file(extension='.tar')})
def setup(opts):
    # 1. load pre-tained model
    checkpoint_fp = opts['checkpoint']
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    return model.eval()



@runway.command('classify', inputs={'photo': image}, outputs={'image': image})
def classify(model, inputs):
    in_img = inputs['photo']
    img_ori = np.array(in_img)
    img_fp = 'samples/test1.jpg'


    face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    #print(transform)
    rects = face_detector(img_ori, 1)

    pts_res = []
    Ps = []  # Camera matrix collection
    poses = []  # pose collection, [todo: validate it]
    vertices_lst = []  # store multiple face vertices
    ind = 0
    suffix = get_suffix(img_fp)
    for rect in rects:
        # - use detected face bbox
        bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
        roi_box = parse_roi_box_from_bbox(bbox)

        img = crop_img(img_ori, roi_box)

        # forward: one step
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0)
        print(input)
        with torch.no_grad():
            
            if mode == 'gpu':
                input = input.cuda()

            param = model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)


        # 68 pts
        pts68 = predict_68pts(param, roi_box)

        # two-step for more accurate bbox to crop face
        if bbox_init == 'two':
            roi_box = parse_roi_box_from_landmark(pts68)
            img_step2 = crop_img(img_ori, roi_box)
            img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img_step2).unsqueeze(0)
            with torch.no_grad():
                if mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            pts68 = predict_68pts(param, roi_box)

        pts_res.append(pts68)
        P, pose = parse_pose(param)
        Ps.append(P)
        poses.append(pose)

    
        vertices = predict_dense(param, roi_box)
        vertices_lst.append(vertices)
        ind += 1
        

    pncc_feature = cpncc(img_ori, vertices_lst, tri - 1)
    output = pncc_feature[:, :, ::-1]
    print(type(output))
    pilImg = transforms.ToPILImage()(np.uint8(output))

    return { "image": pilImg } 


if __name__ == '__main__':
    # run the model server using the default network interface and ports,
    # displayed here for convenience
    runway.run(host='0.0.0.0', port=8000)