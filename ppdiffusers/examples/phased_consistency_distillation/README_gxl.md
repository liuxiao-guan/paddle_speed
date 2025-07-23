## 交接
#### conda 环境： gxl_zx （在a800上已有）
#### 整个流程
1. 下载数据集程序：/root/paddlejob/workspace/env_run/test_data/coco1k/download.py 
2. 下载解压模型 
```shell
下载解压模型脚本：/root/paddlejob/workspace/env_run/test_data/down.sh  # wget XXX.tar.gz
现有已下载的模型在： /root/paddlejob/workspace/env_run/test_data
```
3. 生成图片：如下

```shell
eval_infer_pcm_flux_coco1k.py ## 用来跑lastest checkpoint的模型
具体参考脚本：scripts/coco1k.sh

eval_infer_pcm_flux_coco1k_subset.py  ## 用来跑一个模型的不同checkpoint的 主要修改文件命名 加入了checkpoint_version的区分
具体参考脚本：scripts/coco1k_subset.sh
eval_infer_pcm_flux_coco10k.py ## 用来跑coco10k数据集 还没有脚本  记得修改生成图片的文件名

```
4. 保存图片
保存图片的地址：/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_coco1k
5. 计算fid与clip score

#### 其余流程
可视化代码：就在生成图片的地方 /root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_coco1k/vis_prompt_coco1k.py   
想要可视化那个文件，直接在file_list = [] 里面添加文件名即可
然后在当前目录会生成文件名.html 文件 
```shell
查找服务器地址：ip a  
如：10.174.146.88 
开启一个服务端口：python -m http.server 8000
可视化链接：http://10.174.146.88:8000/文件名.html # 尽量不要用localhost 北京那边的同事点不开 
```

 


