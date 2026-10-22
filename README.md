无人机垃圾堆检测数据集 -![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e83d0e17e60541b6832edade80e81011.png)
航拍图像垃圾倾倒检测数据集，2115张，yolo，voc，coco三种标注方式
图像尺寸:1280*1280
类别数量:1类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4084995c9f8348b3b7e58826f45d42db.png)

训练集图像数量:1476; 验证集图像数量:414； 测试集图像数量:225
类别名称: 每一类图像数 ，每一类标注数
waste: 1981,12840
image num: 2115![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9a8914de21a3482184d1c1271ae8fe2d.png)
1![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/190096985a4d4dadabbc78481d11f7a7.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/21d18c48d1314b729bacac7ef22f6f6f.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/51d5d868107043c3b0a8c55807a2dbcd.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/29a78e8fabe64635b781cef1cf71e65e.png)
# 航拍图像垃圾倾倒检测数据集
|项目|参数|
| ---- | ---- |
|数据集名称|航拍图像垃圾倾倒检测数据集|
|总图像|2115张|
|图像尺寸|1280×1280|
|标注格式|YOLO、VOC、COCO|
|类别数|1类|
|训练集|1476张|
|验证集|414张|
|测试集|225张|

|类别|中文|图像数|标注框总数|
| ---- | ---- | ---- | ---- |
|waste|垃圾倾倒堆|1981|12840|

## dataset.yaml
```yaml
path: ./aerial_waste
train: images/train
val: images/val
test: images/test

names:
  0: waste
```

## train.py
```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11s.pt")
    model.train(
        data="dataset.yaml",
        epochs=85,
        imgsz=1280,
        batch=4,
        device=0,
        patience=10,
        cos_lr=True,
        mosaic=0.75,
        mixup=0.1,
        project="runs/train",
        name="aerial_waste_detect"
    )
```

## predict.py
```python
from ultralytics import YOLO

model = YOLO("./runs/train/aerial_waste_detect/weights/best.pt")
results = model.predict(
    source="./test_images",
    imgsz=1280,
    conf=0.2,
    iou=0.45,
    save=True
)
```

### 环境依赖
```bash
pip install ultralytics opencv-python numpy
```

