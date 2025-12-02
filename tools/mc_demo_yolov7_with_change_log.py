import argparse
import time
from pathlib import Path
import sys
import json
from datetime import datetime
from collections import Counter
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

sys.path.insert(0, './yolov7')
sys.path.append('.')

# Global variables to track changes
previous_objects = {}
change_log = []

def log_object_change(frame_num, timestamp, current_objects, video_fps):
    """Log changes in detected objects"""
    global previous_objects, change_log
    
    # Convert frame number to timestamp (MM:SS)
    seconds = frame_num / video_fps if video_fps > 0 else 0
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    time_str = f"{minutes:02d}:{secs:02d}"
    
    # Check if objects changed
    if current_objects != previous_objects:
        # Determine what changed
        changes = []
        
        # Check for new objects
        for obj_class, count in current_objects.items():
            prev_count = previous_objects.get(obj_class, 0)
            if count > prev_count:
                changes.append(f"+{count - prev_count} {obj_class}(s) appeared")
            elif count < prev_count:
                changes.append(f"-{prev_count - count} {obj_class}(s) disappeared")
        
        # Check for objects that completely disappeared
        for obj_class in previous_objects:
            if obj_class not in current_objects:
                changes.append(f"All {obj_class}(s) disappeared")
        
        # Create log entry
        log_entry = {
            "frame": frame_num,
            "timestamp": time_str,
            "total_objects": sum(current_objects.values()),
            "objects": dict(current_objects),
            "changes": changes,
            "log_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        change_log.append(log_entry)
        
        # Print to console
        print(f"\n[CHANGE DETECTED] Frame {frame_num} ({time_str})")
        print(f"  Objects: {dict(current_objects)}")
        print(f"  Changes: {', '.join(changes)}")
        
        # Update previous state
        previous_objects = current_objects.copy()
        
        return True
    
    return False

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    print('save results to {}'.format(filename))


def detect(save_img=False):
    global change_log
    
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # Create tracker
    tracker = BoTSORT(opt, frame_rate=30.0)
    results = []
    frame_id = 0
    
    # Get video FPS for timestamp calculation
    if not webcam:
        cap = cv2.VideoCapture(source)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"\nVideo Info: {total_frames} frames @ {video_fps:.2f} FPS")
        print(f"Duration: {total_frames/video_fps:.2f} seconds\n")
    else:
        video_fps = 30.0
        total_frames = 0

    # Frame skip settings - process 1 frame per second
    frame_skip = int(video_fps) if opt.frame_skip == 1 else opt.frame_skip  # Skip to get 1 fps
    print(f"Frame skip: Processing every {frame_skip} frame(s) = {video_fps/frame_skip:.1f} FPS\n")

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # Skip frames based on frame_skip setting
        if frame_id % frame_skip != 0:
            frame_id += 1
            continue
            
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                object_counts = Counter()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    object_counts[names[int(c)]] = int(n)
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Track objects
                import numpy as np
                detections = []
                for *xyxy, conf, cls in reversed(det):
                    detections.append([float(x) for x in [*xyxy, conf, cls]])
                detections = np.array(detections) if detections else np.empty((0, 6))
                
                online_targets = tracker.update(detections, im0)

                # Log object changes
                log_object_change(frame_id, time.time(), object_counts, video_fps)

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > opt.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # Draw boxes
                        if save_img or view_img:
                            label = f'{tid} {names[int(t.cls)]} {t.score:.2f}'
                            plot_one_box(t.tlbr, im0, label=label, color=colors[int(t.cls)], line_thickness=2)

                results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            else:
                # No detections - log if there were objects before
                if previous_objects:
                    log_object_change(frame_id, time.time(), Counter(), video_fps)
                tracker.update(np.empty((0, 6)), im0)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, Frame {frame_id}/{total_frames if total_frames else "?"}''\r', end='')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

            frame_id += 1

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    # Save change log
    log_file = save_dir / 'object_change_log.json'
    with open(log_file, 'w') as f:
        json.dump(change_log, f, indent=2)
    print(f"\n\n✅ Change log saved to: {log_file}")
    print(f"📊 Total changes detected: {len(change_log)}")
    
    # Also save a human-readable text version
    txt_log_file = save_dir / 'object_change_log.txt'
    with open(txt_log_file, 'w') as f:
        f.write(f"Object Change Log\n")
        f.write(f"Video: {source}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 80 + "\n\n")
        
        for entry in change_log:
            f.write(f"Frame {entry['frame']:5d} | Time {entry['timestamp']} | Total Objects: {entry['total_objects']}\n")
            f.write(f"  Objects: {entry['objects']}\n")
            f.write(f"  Changes: {', '.join(entry['changes'])}\n")
            f.write("-" * 80 + "\n")
    
    print(f"📄 Human-readable log saved to: {txt_log_file}")
    
    print(f'\nDone. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold')
    parser.add_argument('--frame-skip', type=int, default=1, help='process every N frames (1=process 1 fps, 0=process all frames)')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=True, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str,
                        help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')
    
    # Additional arguments
    parser.add_argument("--jde", default=False, action="store_true", help="JDE mode")
    parser.add_argument("--ablation", default=False, action="store_true", help="Ablation mode")

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
