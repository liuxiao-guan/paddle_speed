import paddle
from paddle import nn
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel
# paddle.enable_static()
#     # 保持与之前相同的模型定义...
# # 实例化模型
# model = FluxTransformer2DModel()
# # 定义输入占位符（使用paddle.static.data）
# image_input = paddle.static.data(
#     name='image', 
#     shape=[1, 3, 512, 512], 
#     dtype='float32'
# )
# txt_input = paddle.static.data(
#     name='text', 
#     shape=[1, 77], 
#     dtype='int64'
# )
# timestep = paddle.static.data(
#     name='timestep', 
#     shape=[1], 
#     dtype='int64'
# )
 
# # 创建前向传播函数（关键修正）
# @paddle.jit.to_static
# def forward_func(*args):
#     return model(*args)
 
# # 计算FLOPs（使用新的调用方式）
# flops, params = paddle.flops(
#     forward_func,
#     input_shapes=[  # 新版本参数名
#         [1, 3, 512, 512],  # 图像输入
#         [1, 77],            # 文本输入
#         [1]                 # 时间步
#     ],
#     input_dtypes=['float32', 'int64', 'int64'],  # 显式指定数据类型
#     print_detail=True
# )
 
# print(f"Total FLOPs: {flops / 1e12:.2f} TFLOPs")
# print(f"Total Parameters: {params / 1e6:.2f} M")
 
# # 退出静态图模式
# paddle.disable_static()

# 动态图模式计算（实验性）
model = FluxTransformer2DModel()
# 创建完整输入（动态图模式）
image_input = paddle.randn([1, 3, 512, 512], dtype='float32')  # 图像输入
txt_input = paddle.randint(0, 49408, [1, 77], dtype='int64')    # 文本输入（77 tokens）
timestep = paddle.to_tensor([100], dtype='int64')               # 时间步输入
 
# 计算FLOPs（动态图模式）
flops, params = paddle.flops(
    model,
    input_size=[
        image_input.shape,
    ],
    
    print_detail=True
)
 
print(f"Total FLOPs: {flops / 1e12:.2f} TFLOPs")
print(f"Total Parameters: {params / 1e6:.2f} M")