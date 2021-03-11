import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, output_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_head = num_heads
        self.output_dim = output_dim
        split_output_dim = output_dim // num_heads
        self.dim_k = split_output_dim
        self.dim_q = split_output_dim
        self.dim_v = split_output_dim

        self.v_layer = nn.Linear(self.output_dim, self.dim_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.output_dim, self.dim_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.output_dim, self.dim_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.dim_v, self.output_dim, bias=False)

    def forward(self, q, k, v, mask=None):
        heads_list = []
        attention_list = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            heads_list.append(head)
            attention_list.append(attn)
        stacked_heads = torch.stack(heads_list, dim=2) if self.n_head > 1 else heads_list[0]
        attn = torch.stack(attention_list, dim=2)
        outputs = torch.mean(stacked_heads, dim=2) if self.n_head > 1 else stacked_heads
        outputs = self.w_h(outputs)
        return outputs, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))

        if self.scale:
            dimension = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


