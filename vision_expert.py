import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_patches = (config.img_size // config.patch_size) ** 2
        self.proj = nn.Conv2d(config.in_channels, config.embed_dim, kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, x):
        x = self.proj(x)  
        x = x.flatten(2).transpose(1, 2)  
        return x
    
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.scale = (config.dim // config.num_heads) ** -0.5
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.proj = nn.Linear(config.dim, config.dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))
    
class MixtureOfExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.num_experts)])
        self.gate = nn.Linear(config.dim, config.num_experts)
    
    def forward(self, x):
        gate_scores = self.gate(x).softmax(dim=-1) 
        topk_values, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        B, N, C = x.shape
        out = torch.zeros_like(x)
        
        for expert_idx in range(self.num_experts):
            mask = (topk_indices == expert_idx)
            if mask.sum() == 0:
                continue  
            x_flat = x.view(B * N, C)
            mask_flat = mask.view(B * N, self.top_k)
            
            indices = torch.nonzero(mask_flat, as_tuple=False) 
            
            if indices.numel() == 0:
                continue
            
            token_indices = indices[:, 0]  
            gate_weights = topk_values.view(B * N, self.top_k)[indices[:, 0], indices[:, 1]]
            
            expert_tokens = x_flat[token_indices] 
            expert_output = self.experts[expert_idx](expert_tokens)  
            expert_output = expert_output * gate_weights.unsqueeze(-1)
            
            out_flat = out.view(B * N, C)
            out_flat.index_add_(0, token_indices, expert_output)
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = MixtureOfExperts(config) if config.use_moe else MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch_embed.num_patches, config.dim))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)  
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x[:, 0])  
        return self.head(x)

@dataclass
class Config():
    img_size=32
    patch_size=4
    in_channels=3
    embed_dim=768
    dropout=0.1
    num_experts=4
    top_k=2
    dim=768
    hidden_dim = dim * 4
    num_classes=100
    depth=6
    num_heads=6
    mlp_ratio=4.0
    use_moe=True

