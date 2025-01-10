# Qwen2-VL

## 1. 模型介绍

[Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/) 是 Qwen 团队推出的一个专注于视觉与语言（Vision-Language, VL）任务的多模态大模型。它旨在通过结合图像和文本信息，提供强大的跨模态理解能力，可以处理涉及图像描述、视觉问答（VQA）、图文检索等多种任务。Qwen2-VL通过引入创新性的技术如 Naive Dynamic Resolution 和 M-RoPE，以及深入探讨大型多模态模型的潜力，显著地提高了多模态内容的视觉理解能力。

## 2 环境准备

- **python >= 3.10**
- **paddlepaddle-gpu 要求是develop版本**
```bash
# 安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

- **paddlenlp 需要特定版本**

在PaddleMIX/代码目录下执行以下命令安装特定版本的paddlenlp：
```bash
# 安装示例
git submodule update --init --recursive
cd PaddleNLP
git reset --hard e91c2d3d634b12769c30aa419ddf931c20b7ca9f
pip install -e .
cd csrc
python setup_cuda.py install
```

> 注：
* 请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
* (默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡

## 3 高性能推理

在Qwen2-VL的高性能推理优化中，**视觉模型部分继续使用PaddleMIX中的模型组网；但是语言模型部分调用PaddleNLP中高性能的Qwen2语言模型**，以得到高性能的Qwen2-VL推理版本。

### 3.1. 文本&单张图像输入高性能推理
```bash
CUDA_VISIBLE_DEVICES=0 python deploy/qwen2_vl/single_image_infer.py \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dtype bfloat16 \
    --benchmark True \
```

- 在 NVIDIA A800-80GB 上测试的单图端到端速度性能如下：

| model                  | Paddle Inference|    PyTorch   | Paddle 动态图 |
| ---------------------- | --------------- | ------------ | ------------ |
| Qwen2-VL-2B-Instruct   |      1.053 s     |     2.086 s   |   5.766 s   |
| Qwen2-VL-7B-Instruct   |      2.293 s     |     3.132 s   |   6.221 s   |


### 3.2. 文本&视频输入高性能推理
```bash
CUDA_VISIBLE_DEVICES=0 python deploy/qwen2_vl/video_infer.py \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dtype bfloat16 \
    --benchmark True
```

- 在 NVIDIA A800-80GB 上测试的单视频端到端速度性能如下：

| model                  | Paddle Inference|    PyTorch   | Paddle 动态图 |
| ---------------------- | --------------- | ------------ | ------------ |
| Qwen2-VL-2B-Instruct   |      2.890 s    |     3.143 s  |    6.183 s  |
| Qwen2-VL-7B-Instruct   |      2.534 s    |     2.715 s  |    5.721 s  |
