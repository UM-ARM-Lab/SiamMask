import os
import glob
import sys
from SiamMask.tools.test import *
from SiamMask.experiments.siammask_sharp.custom import Custom
import argparse
import torch
from mps_msgs.msg import AABBox2d

def demo_ros(bbox, ims):
    dirpath = os.getcwd()
    # print("current directory is : " + dirpath)
    os.chdir("/home/kunhuang/catkin_ws/src/SiamMask/src/SiamMask/experiments/siammask_sharp")

    dirpath = os.getcwd()
    # print("current directory is : " + dirpath)
    sys.path.insert(1, '/home/kunhuang/catkin_ws/src/SiamMask/src/SiamMask/experiments/siammask_sharp/')
    sys.path.insert(1, '/home/kunhuang/catkin_ws/src/SiamMask/src/SiamMask/tools')
    sys.path.insert(1, '/home/kunhuang/catkin_ws/src/SiamMask/src/SiamMask')
    # print(sys.path)


    parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

    parser.add_argument('--resume', default='', type=str, required=True,
                        metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--config', dest='config', default='config_davis.json',
                        help='hyper-parameter of SiamMask in json format')
    parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
    parser.add_argument('--cpu', action='store_true', help='cpu mode')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    # img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    # # ims is a sequence of cv2 images, does the type accord with cv_bridge::CvImagePtr????
    # ims = [cv2.imread(imf) for imf in img_files] 

    # Select ROI
    x = bbox.xmin
    y = bbox.ymin
    w = bbox.xmax - bbox.xmin
    h = bbox.ymax - bbox.ymin
    # cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # try:
    #     init_rect = cv2.selectROI('SiamMask', ims[0], False, False) # This is the step for setting bounding box
    #     x, y, w, h = init_rect # All information from inputted ROI is {x, y, w, h}
    # except:
    #     exit()

    maskVideo = []
    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            print(type(mask))
            # print(mask.shape)
            # print(mask.astype(float))

            # im.setflags(write=1)
            im_out = im.copy()
            im_out[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            mask_im = im.copy()

            cv2.polylines(im_out, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', im_out)
            # cv2.imshow('SiamMask', mask.astype(float))
            
            maskVideo.append(np.uint8(mask))

            key = cv2.waitKey(100)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
    return maskVideo


def demo_ros_subscriber(bbox):
    dirpath = os.getcwd()
    print("current directory is : " + dirpath)
    os.chdir("/home/kunhuang/catkin_ws/src/SiamMask/src/SiamMask/experiments/siammask_sharp")

    dirpath = os.getcwd()
    print("current directory is : " + dirpath)
    sys.path.insert(1, '/home/kunhuang/catkin_ws/src/SiamMask/src/SiamMask/experiments/siammask_sharp/')
    sys.path.insert(1, '/home/kunhuang/catkin_ws/src/SiamMask/src/SiamMask/tools')
    sys.path.insert(1, '/home/kunhuang/catkin_ws/src/SiamMask/src/SiamMask')
    # print(sys.path)


    parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

    parser.add_argument('--resume', default='', type=str, required=True,
                        metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--config', dest='config', default='config_davis.json',
                        help='hyper-parameter of SiamMask in json format')
    parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
    parser.add_argument('--cpu', action='store_true', help='cpu mode')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    # ims is a sequence of cv2 images, does the type accord with cv_bridge::CvImagePtr????
    ims = [cv2.imread(imf) for imf in img_files] 

    # Select ROI
    x = bbox.xmin
    y = bbox.ymin
    w = bbox.xmax - bbox.xmin
    h = bbox.ymax - bbox.ymin
    # cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # try:
    #     init_rect = cv2.selectROI('SiamMask', ims[0], False, False) # This is the step for setting bounding box
    #     x, y, w, h = init_rect # All information from inputted ROI is {x, y, w, h}
    # except:
    #     exit()

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', im)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
