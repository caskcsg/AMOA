# AMOA: Global Acoustic Feature Enhanced Modal-Order-Aware Network for Multimodal Sentiment Analysis

从以下链接下载MOSI数据集，解压后将文件夹放在根目录下。

链接：https://pan.baidu.com/s/1-Ok9LJQaXUEf4xxBSVEuFw 
提取码：5ryu 

从以下链接下载MOSEI数据集，解压后将文件夹放在根目录下。

链接：https://pan.baidu.com/s/19XsM9CNGMwuAdMQ8-_SSUw 
提取码：d65m 

从以下链接下载BERT相关文件，解压后将文件夹放在根目录下。

链接：https://pan.baidu.com/s/1e63OggGwvOb9bsf-3OkSwQ 
提取码：tsiu 

代码运行：

```
python run.py --dataset MOSI --contrastive
```

其中--dataset参数用来选择数据集，--contrastive参数用来选择是否使用对比学习。其他参数详见run.py文件，可根据自己需要进行调整。

