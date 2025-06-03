
### generation.py
#### 该文件是生成图片的代码，
```sh
# 运行实例
CUDA_VISIBLE_DEVICES=3 nohup python generation.py \
--inference_step 50 \  #推导生成的步数
--seed 124 \   # 随机种子
--dataset 'coco1k' \    ## 提取prompt的来源数据集
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \    ## 获取prompt list的位置
--teacache > output.log 2>&1 &    ## 使用的推理加速方法 如果是没有使用任何加速的就是 --origin   使用tgate的就是 --tgate 
```

#### 如果想加入自己的方法 直接在末尾再添加一个if语句就可以 
所有的图片的父目录都保存在```/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed``` 根据不同的方法创建不同的子目录
#### 如果是想加入SD3模型的对应方法 那就需要改if里面的命令 因为里面默认是fluxpipeline  
注意⚠️：SD3模型生成的方法 一定要改掉生成的子目录的文件名 因为不改的话会直接覆盖原先flux生成的 代码里具体指```saved_path``` 变量


### evalution.py 
#### 该文件是生成图片的代码 主要是分析FID, PSNR, SSIM
```sh
# 运行实例
CUDA_VISIBLE_DEVICES=2  nohup python evaluation.py \
--inference_step 50 \     ## 推导步数
--seed 124 \              ## 随机种子
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \  ##coco1k的原图片地址
--generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/origin_50steps_coco1k \   ## 原始方法生成的图片地址
--speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/teacache_coco1k \   ## teacache方法生成的图片地址
--resolution 1024 > output_1.log 2>&1 &  ## 设置分辨率
```
#### 一般来说 如果是加入自己的方法一般改一下```speed_generation_path```的路径就可以

### 运行的conda 环境为paddlemix
