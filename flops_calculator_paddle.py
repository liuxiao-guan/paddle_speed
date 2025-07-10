import paddle
import paddle.nn as nn
from ppdiffusers import FluxTransformer2DModel
from paddle.utils import flops
paddle.flops
# 创建模型实例
model = FluxTransformer2DModel(
    patch_size=1,
    in_channels=64,
    num_layers=19,
    num_single_layers=38,
    attention_head_dim=128,
    num_attention_heads=24
)

# 定义输入
hidden_states = paddle.randn([1, 256, 64])  # batch_size=1, seq_len=256, in_channels=64
timestep = paddle.to_tensor([0.5])
pooled_projections = paddle.randn([1, 768])
txt_ids = paddle.zeros([256, 3])
img_ids = paddle.zeros([256, 3])

# 计算FLOPs
hidden_states = paddle.randn([1, 256, 64])  # batch_size=1, seq_len=256, in_channels=64
timestep = paddle.randn([1])                # timestep
pooled_projections = paddle.randn([1, 768]) # pooled_projections
txt_ids = paddle.randn([256, 3])            # txt_ids
img_ids = paddle.randn([256, 3])            # img_ids
# 计算FLOPs需要指定输入形状而不是实际张量
input_shapes = [
    [1, 256, 64],  # hidden_states shape
    [1],           # timestep shape
    [1, 768],      # pooled_projections shape
    [256, 3],      # txt_ids shape
    [256, 3]       # img_ids shape
]
# 创建完整的输入参数
hidden_states = paddle.randn(input_shapes[0]).astype('float32')
encoder_hidden_states = paddle.randn([1, 256, 4096]).astype('float32')  # 添加encoder_hidden_states
timestep = paddle.randn(input_shapes[1]).astype('float32')
pooled_projections = paddle.randn(input_shapes[2]).astype('float32')
txt_ids = paddle.randint(0, 1000, input_shapes[3]).astype('int64')
img_ids = paddle.randint(0, 1000, input_shapes[4]).astype('int64')

# 直接调用模型forward方法计算FLOPs
output = model(
    hidden_states=hidden_states,
    encoder_hidden_states=encoder_hidden_states,  # 添加encoder_hidden_states参数
    timestep=timestep,
    pooled_projections=pooled_projections,
    txt_ids=txt_ids,
    img_ids=img_ids
)

# 使用paddle.flops计算FLOPs
flops_count = paddle.flops(
    model,
    input_shapes[0],  # 只传递第一个输入的shape，因为paddle.flops只接受单个input_size
    custom_ops=None,
    print_detail=False,
    inputs=[hidden_states, encoder_hidden_states, timestep, pooled_projections, txt_ids, img_ids]
)
print(f"FLOPs count: {flops_count/1e9:.2f} GFLOPs")