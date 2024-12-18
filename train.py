import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolo11-C3k2-SHSA-CGLU.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='VOC.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=128,
                close_mosaic=0,
                workers=4, 
                device='0',
                optimizer='SGD', # using SGD
                patience=0, # set 0 to close earlystop.
                # resume=True, 
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='yolo11-C3k2-SHSA-CGLU',
                )