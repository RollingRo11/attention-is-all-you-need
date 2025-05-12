# Transformer

Python implementation of "Attention Is All You Need" (Vaswani et al.)

Read the paper [here!](https://arxiv.org/abs/1706.03762)

Restructured and inspired heavily from Andrej Karpathy's [GPT Video!](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6424s)

## Paper implementation overview

The key innovation of the Transformer is the self-attention mechanism, which allows the model to weigh the importance of different positions in the input sequence when computing a representation of the sequence.

**Architecture Overview:**
- Decoder-only structure with stacked self-attention and fully connected layers
- Multi-head attention allows the model to jointly attend to information from different representation subspaces
- Position-wise feed-forward networks
- Positional encodings to incorporate sequence order information

## Key Mathematical Components

### Scaled Dot-Product Attention

The attention function maps queries and keys of dimension $d_k$, and values of dimension $d_v$ to an output:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The scaling factor of $\frac{1}{\sqrt{d_k}}$ is used to prevent the dot products from growing too large in magnitude, pushing the softmax function into regions with extremely small gradients.

Implemented like so:
```Python
wei = q @ k.transpose(-2, -1) * (C**-0.5)
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
wei = F.softmax(wei, dim=1)
wei = self.dropout(wei)
```

### Multi-Head Attention

Instead of performing a single attention function, the model linearly projects queries, keys, and values $h$ times with different learned projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$


## Implementation

The most important implementation is the Multihead Attention, found alongside all of the other most relevant code in [`layers.py`](https://github.com/RollingRo11/attention-is-all-you-need/blob/main/layers.py).

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
```



## Key Components

- **Encoder Stack**: 6 identical layers with multi-head self-attention and position-wise feed-forward networks
- **Decoder Stack**: 6 identical layers with an additional layer for encoder-decoder attention
- **Position Encodings**: Sine and cosine functions of different frequencies
- **Layer Normalization**: Applied after each sub-layer
- **Residual Connections**: Around each sub-layer
- **Dropout**: Applied for regularization with rate $P_{drop} = 0.1$

The implementation here follows the architecture described in the original paper, with the model parameters set to match those used in the experiments: $N = 6$, $d_{\text{model}} = 512$, $d_{ff} = 2048$, $h = 8$, $d_k = d_v = 64$.
