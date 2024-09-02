import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import torch.nn.functional as F
def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

# https://github.com/tencent-ailab/IP-Adapter/issues/54
# import cv2                                                                                                        
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
# from insightface.utils import face_align

# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
# img = cv2.imread("person.png")

# faces = app.get(img)
# norm_face = face_align.norm_crop(img, landmark=faces[0].kps, image_size=224)

import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, f'd_model={d_model}, numheads={num_heads}'
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        Q = self.split_heads(self.W_q(query), batch_size)
        K = self.split_heads(self.W_k(key), batch_size)
        V = self.split_heads(self.W_v(value), batch_size)

        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim**0.5)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        x = torch.matmul(attention_weights, V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        x = self.W_o(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attention_output = self.multi_head_attention(x, x, x)
        x = self.layer_norm1(x + attention_output)
        feed_forward_output = self.feed_forward(x)
        x = self.layer_norm2(x + feed_forward_output)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([TransformerLayer(d_model, num_heads) for _ in range(num_layers)])
    
    def forward(self, x):
        x = self.embedding(x)
        for _ in range(self.num_layers):
            x = self.layers[_](x)
        return x

# Example usage:
# input_dim = 512  # Dimension of the input tensor
# num_heads = 8    # Number of attention heads
# num_layers = 3   # Number of transformer layers

# # Create an instance of the Transformer model
# model = Transformer(input_dim, num_heads, num_layers)

# # Test the model with a random input tensor (batch_size, sequence_length, d_model)
# batch_size, sequence_length = 16, 20
# input_tensor = torch.randn(batch_size, sequence_length, input_dim)
# output = model(input_tensor)

# print("Input shape:", input_tensor.shape)
# print("Output shape:", output.shape)


