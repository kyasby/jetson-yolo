import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general  import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.utils.tools  import objid2name, objname2id
from deep_sort_pytorch.deep_sort import DeepSort

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import tqdm
import datetime
import csv


palette = (2 ** 5 - 1, 2 ** 15 - 1, 2 ** 20 - 1)



def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top  = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top  + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, obj_id=0, is_detected = False, direction=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        ID     = int(identities[i]) if identities is not None else 0
        color  = (0,255,0) if is_detected[i] else compute_color_for_labels(int(obj_id[i]))

        label  = '{}{:d}'.format("", ID)
        name = objid2name(obj_id[i])

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2),                          color,  3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0], y1 + t_size[1] ), color, -1)

        cv2.putText(img, label, (x1, y1 + t_size[1]),  cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)
        cv2.putText(img, name,  (x1, y1 + t_size[1]+20), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)
        cv2.putText(img, str(direction[i]),  (x1, y1 + t_size[1] + 40), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)

    return img


# here ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def draw_count(img, count, classes):
    idx = 1
    for name, num in count.items():
        obj_id = objname2id(name)
        if obj_id in classes:
            cv2.putText(img, f"{name}: left:{num['left']}, right:{num['right']}", (0, (idx)*30),  cv2.FONT_HERSHEY_PLAIN, 2, [255,0,0], 2)
            idx += 1
    return img
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def cover_mask(mask_img_file_path, image, shape):
    img = cv2.imread(mask_img_file_path)

    img = cv2.resize(img, dsize=(shape[1],shape[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray < 200, 0, 255).astype(np.uint8)
    mask = np.stack([mask, mask, mask], axis=2)
    mask = mask.transpose(2,0,1)

    return np.where(mask == 255, image, 0)

    

def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz, road_direction, mask_img_file_path = (opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt,
                                                                                             opt.img_size, opt.road_direction, opt.mask_img_file_path)
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, n_count=cfg.DEEPSORT.N_COUNT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True,
                        road_direction=road_direction)
    

    # Initialize
    device = select_device(opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    os.makedirs(out, exist_ok=True)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # output csv
    dt_today = datetime.date.today()
    minute_last = datetime.datetime.now()
    dt_minute = minute_last.strftime("%Y/%m/%d_%H:%M:%S")
    
    df = pd.DataFrame(deepsort.count).reset_index()
    df_date = pd.DataFrame([dt_minute, dt_minute])
    df = df.merge(df_date, left_index=True, right_index=True)
    df.to_csv(f"./{dt_today}.csv", index=False)

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset  = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0  = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path  = str(Path(out)) + '/results.txt'
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        if mask_img_file_path:
            img = cover_mask(mask_img_file_path, img, img.shape[1:])
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        


        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs     = []
                # here~~~~~~~~~~~~~~~~~~~~~
                classes   = []
                # ~~~~~~~~~~~~~~~~~~~~~~~~~

                # Adapt detections to deep sort input format
                for *xyxy, conf, obj_cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    # here ~~~~~~~~~~~~~~~~~~~
                    if (bbox_w * bbox_h) > 1500:
                    # ~~~~~~~~~~~~~~~~~~~~~~~~
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                        # here ~~~~~~~~~~~~~~~~~~~~~~~
                        classes.append(obj_cls.item())
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


                xywhs    = torch.Tensor(bbox_xywh)
                confss   = torch.Tensor(confs)
                # here ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                classess = np.array(classes, dtype=np.uint8)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # Pass detections to deepsort

                outputs = deepsort.update(xywhs, confss, classes, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy  = outputs[:, :4]
                    
                    # here ~~~~~~~~~~~~~~~~~~~~~~
                    direction   = outputs[:, -4]
                    identities  = outputs[:, -3]
                    obj_id      = outputs[:, -2]
                    is_detected = outputs[:, -1]

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    draw_boxes(im0, bbox_xyxy, identities, obj_id, is_detected, direction)

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        # here -1 => -3 ~~~~~~~~
                        identity = output[-3]
                        # ~~~~~~~~~~~~~~~~~~~~~~
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            im0 = draw_count(im0, deepsort.count, opt.classes)

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # here
            if mask_img_file_path:
                im0 = im0.transpose(2,0,1)
                im0 = cover_mask(mask_img_file_path, im0, im0.shape[1:])
                im0 = im0.transpose(1,2,0)
            

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)


        minute_tmp = datetime.datetime.now().minute
        if minute_tmp != minute_last:
            minute_last = minute_tmp
            df = pd.DataFrame(deepsort.count).reset_index()
            dt_minute = datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
            df_date = pd.DataFrame([dt_minute, dt_minute])
            df = df.merge(df_date, left_index=True, right_index=True)
            df.to_csv(f"./{dt_today}.csv", mode="a", header=False, index=False)

    if save_txt or save_img:
        print('Results saved to %s' % out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    cap = cv2.VideoCapture(source)
    video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # フレーム数を取得する
    video_fps = cap.get(cv2.CAP_PROP_FPS)                 # フレームレートを取得する
    video_len_sec = video_frame_count / video_fps         # 長さ（秒）を計算する
    length = datetime.timedelta(seconds=video_len_sec)
    print(f"動画時間：{str(length).split('.')[0]}\n")

    final_count = deepsort.count
    for muki, direction in zip(["右向き", "左向き"], ["right", "left"]):
        result_text =  f"{muki}\n"
        result_text += f"小型：{final_count['car'][direction]}台\n"
        result_text += f"大型：{final_count['bus'][direction] + final_count['truck'][direction]}台\n"
        result_text += f"二輪：{final_count['motorcycle'][direction]}台\n"
        print(result_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5x.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='full path to output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[2, 3, 5, 7], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--road-direction", type=str,
                        default="", help="right_up or right_down")
    parser.add_argument("--mask-img-file-path", type=str,
                        default="", help="full path to mask image_ ile")



    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    
    assert (args.road_direction == "left_is_up" or args.road_direction == "left_is_down"), f"expacted value for road_direction is 'left_is_up' or 'left_is_down', but input one was '{args.road_direction}'" 
    assert (args.output != ""), f"designate out put path" 
    print(args)

    with torch.no_grad():
        detect(args)
