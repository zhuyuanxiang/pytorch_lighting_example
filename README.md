<div align="center">    
# PyTorch Lightning Examples

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>

## 描述   
这是一个 pytorch lightning 的样例仓库，里面有帮助文件中的样例，也有工作需要使用的模型。   

## 如何运行
首先，安装依赖

```bash
# clone project   
git clone https://github.com/zhuyuanxiang/pytorch_lighting_example

# install project   
cd pytorch_lighting_example 
pip install -e .   
pip install -r requirements.txt
```

然后，浏览任意文件，并且运行它。

```bash
# 项目文件夹
cd project

# 运行模型
python train_classifier.py    
```

## 导入

项目作为包的设置可以轻松地导入任意的文件，如下：

```python
from project.torch_datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# datasets
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation

```txt
@article{Zhuyx,
  title={Pytorch-Lightning Examples},
  author={zhuyuanxiang},
  journal={China},
  year={2022}
}
```
