"""
=================================================
@path   : pytorch_lighting_example -> transformer
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/18 15:52
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
https://blog.csdn.net/zhaohongfei_358/article/details/126019181
Pytorch中 nn.Transformer的使用详解与Transformer的黑盒讲解
@Desc   :
@History:
@Plan   :
==================================================
"""
import random

import torch
import torch.nn as nn

from examples.pytorch.transformer.transformer_model import TransformerModel


def generate_random_batch(batch_size, max_length=16):
    """生成随机数据

    Args:
        batch_size (_type_): _description_
        max_length (int, optional): _description_. Defaults to 16.

    Returns:
        _type_: _description_
    """
    src = []
    for i in range(batch_size):
        # 随机生成句子长度
        random_len = random.randint(1, max_length - 2)
        # 随机生成句子词汇，并在开头和结尾增加<bos>和<eos>
        random_nums = [0] + [random.randint(3, 9) for _ in range(random_len)] + [1]
        # 如果句子长度不足max_length，进行填充
        random_nums = random_nums + [2] * (max_length - random_len - 2)
        src.append(random_nums)
    src = torch.LongTensor(src)
    # tgt不要最后一个token
    tgt = src[:, :-1]
    # tgt_y不要第一个的token
    tgt_y = src[:, 1:]
    # 计算tgt_y，即要预测的有效token的数量
    n_tokens = (tgt_y != 2).sum()

    # 这里的n_tokens指的是我们要预测的tgt_y中有多少有效的token，后面计算loss要用
    return src, tgt, tgt_y, n_tokens


def main():
    max_length = 16
    model = TransformerModel()
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    total_loss = 0

    for step in range(2000):
        # 生成数据
        src, tgt, tgt_y, n_tokens = generate_random_batch(batch_size=2, max_length=max_length)

        # 清空梯度
        optimizer.zero_grad()
        # 进行transformer的计算
        out = model(src, tgt)
        # 将结果送给最后的线性层进行预测
        out = model.predictor(out)
        """
        计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
                我们的out的Shape为(batch_size, 词数, 词典大小)，view之后变为：
                (batch_size*词数, 词典大小)。
                而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
                除以n_tokens。
        """
        loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) / n_tokens
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        total_loss += loss

        # 每40次打印一下loss
        if step != 0 and step % 40 == 0:
            print("Step {}, total_loss: {}".format(step, total_loss))
            total_loss = 0
            pass
        pass
    model = model.eval()

    # 随便定义一个src
    src = torch.LongTensor([[0, 4, 3, 4, 6, 8, 9, 9, 8, 1, 2, 2]])
    # tgt从<bos>开始，看看能不能重新输出src中的值
    tgt = torch.LongTensor([[0]])

    # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
    for i in range(max_length):
        # 进行transformer计算
        out = model(src, tgt)
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        predict = model.predictor(out[:, -1])
        # 找出最大值的index
        y = torch.argmax(predict, dim=1)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)

        # 如果为<eos>，说明预测结束，跳出循环
        if y == 1:
            break
    print(tgt)
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # print(generate_random_batch(batch_size=2, max_length=6))
    main()
