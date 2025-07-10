def calculate_transformer_flops(
    batch_size,
    seq_len,
    hidden_dim,
    num_heads,
    num_layers,
    ffn_ratio=4.0
):
    """
    Calculate FLOPs for a transformer model
    
    Args:
        batch_size: int
        seq_len: int
        hidden_dim: int
        num_heads: int
        num_layers: int
        ffn_ratio: float, ratio of hidden dim in FFN
        
    Returns:
        total_flops: float
    """
    # Attention FLOPs
    head_dim = hidden_dim // num_heads
    qkv_flops = 3 * batch_size * seq_len * hidden_dim * head_dim * num_heads
    attn_scores_flops = batch_size * num_heads * seq_len * seq_len * head_dim
    proj_flops = batch_size * seq_len * hidden_dim * hidden_dim
    
    # FFN FLOPs
    ffn_hidden_dim = int(hidden_dim * ffn_ratio)
    ffn_flops = 2 * batch_size * seq_len * hidden_dim * ffn_hidden_dim
    
    # Total per layer
    layer_flops = qkv_flops + attn_scores_flops + proj_flops + ffn_flops
    
    # Total for all layers
    total_flops = layer_flops * num_layers
    
    return total_flops

# Example usage for FluxTransformer
if __name__ == "__main__":
    # From FluxTransformer2DModel config
    hidden_dim = 24 * 128  # num_heads * head_dim
    num_heads = 24
    num_layers = 19 + 38  # transformer_blocks + single_transformer_blocks
    
    # Assuming typical values
    batch_size = 1
    seq_len = 256
    
    total_flops = calculate_transformer_flops(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    print(f"Estimated FLOPs per forward pass: {total_flops/1e9:.2f} GFLOPs")