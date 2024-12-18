import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/train/yolov82/weights/best.pt') 
    model.val(data='VOC.yaml',
              split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=128,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='yolov8',
              )