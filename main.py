from ultralytics import YOLO

# Paper doing a lot of what I want
# https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch
# Table Tennis Dataset: https://lab.osai.ai/

# Another paper using Fast R-CNN
# https://www.nature.com/articles/s41598-024-51865-3#Sec13

model = YOLO("yolo11n.pt")

# device=mps if on Macbook M1
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")

res = model.predict(
    source='pingpong.PNG',
    conf=0.25
)
