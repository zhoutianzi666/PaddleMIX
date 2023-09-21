





- infer_static.py是静态图的推理代码
- 你需要导出两个模型，才可以使用静态图推理




- 为了静态图推理，你需要安装
    - develop版本的Paddle，cuda 需要至少11.2
    - PaddleNLP develop版本


## 第一个模型导出



- 进入PaddleMIX/paddlemix/examples/blip2目录，运行python3.8 export.py即可导出模型，默认在当前的目录下，文件夹名字为blip2_export/


## 第二个模型导出

你需要拉取下PaddleNLP的代码，首先进入PaddleNLP/csrc/目录，执行 python3.8 setup_cuda.py 安装自定义算子。

然后进去目录PaddleNLP/llm，然后执行命令

python3.8  export_model.py      --model_name_or_path /root/.paddlenlp/models/facebook/opt-2.7b      --output_path /zhoukangkang/2023-06-06minigpt/whole_part/opt-2.7b-kaiyuan/      --dtype float16 --inference_model  --model_prefix=opt --model_type=opt-img2txt

即可导出第二部分模型。以上命令你需要更改为自己的路径。


## 运行推理

- 指定了 --first_model_path和 --second_model_path之后你就可以推理了，直接运行python3.8  infer_static.py


