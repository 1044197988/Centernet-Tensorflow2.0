# Centernet-Tensorflow2.0
__Tensorflow2.0下运行目标检测网络Centernet（基于see--的keras-centernet）__<br>
Centernet目标检测属于anchor-free系列的目标检测，相比于CornerNet做出了改进，使得检测速度和精度相比于one-stage和two-stage的框架都有不小的提高，尤其是与YOLOv3作比较，在相同速度的条件下，CenterNet的精度比YOLOv3提高了4个左右的点。<br>

![Centernet](https://github.com/1044197988/Centernet-Tensorflow2.0/blob/master/image/Centernet.png)
<br>
CenterNet是一种用于所有类型的对象检测相关任务的元算法。官方代码解决了2D检测，3D检测和人体姿势估计。对象不是常用的锚盒，而是表示为点。CenterNet还删除了以前单次探测器所需的许多超参数和概念：

* 没有更多的锚箱<br>
- 只有一个功能图表示所有比例<br>
* 没有边界框匹配<br>
- 没有非最大抑制<br>

![Centernet](https://github.com/1044197988/Centernet-Tensorflow2.0/blob/master/image/compare.png)
<br>
![Centernet](https://github.com/1044197988/Centernet-Tensorflow2.0/blob/master/image/image1.png)
<br>
![Centernet](https://github.com/1044197988/Centernet-Tensorflow2.0/blob/master/image/image2.png)
<br>

权重下载：
--- 
点击网址或复制网址到浏览器链接即可下载,比较大,都在700MB以上.<br>
对象检测：<br>
https://github.com/see--/keras-centernet/releases/download/0.1.0/ctdet_coco_hg.hdf5<br>
人体姿态估计：<br>
https://github.com/see--/keras-centernet/releases/download/0.1.0/hpdet_coco_hg.hdf5

## 参考
[see--/keras-centernet](https://github.com/see--/keras-centernet)<br>
感谢see--的代码，修改很少，修改后可以在Tensorflow2.0下运行，在1版本包含keras下，也可以运行。

## 其他框架下的Centernet
[xingyizhou-CenterNet-pytorch-官方](https://github.com/xingyizhou/CenterNet)<br>
[Duankaiwen-CenterNet-pytorch](https://github.com/Duankaiwen/CenterNet)<br>

### 运行:
```Python
python TF2-CenterNet/ctdet_image.py    #对象检测-图片
python TF2-CenterNet/hpdet_image.py    #人体姿态估计-图片
python TF2-CenterNet/ctdet_video.py    #对象检测-视频
python TF2-CenterNet/hpdet_video.py    #人体姿态估计-视频
```
#在coco2017val数据集上执行测试-对象检测：<br>
python TF2-CenterNet/ctdet_coco.py --data val2017 --annotations annotations <br>
#在coco2017val数据集上执行测试-人体姿态估计：<br>
python TF2-CenterNet/hpdet_coco.py --data val2017 --annotations annotations 


