import torch
import torch.nn as nn
import torch.nn.functional as F

# Author: Mohsen Mozafari
# This code implements a simplified version of the GPT model (Generative Pre-trained Transformer),
# including a self-attention mechanism and a feed-forward neural network. The model can be used
# for natural language processing tasks such as text generation.
block_size = 64
n_embd = 384
n_layer = 8
n_heads = 8

class SelfAttention(nn.Module):
    """
    A module that implements the self-attention mechanism along with a feed-forward network.
    
    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        dropout (float): Dropout rate for regularization.
    
    Attributes:
        key_proj (nn.Linear): Linear layer for projecting input to key vectors.
        query_proj (nn.Linear): Linear layer for projecting input to query vectors.
        value_proj (nn.Linear): Linear layer for projecting input to value vectors.
        attention_mask (torch.Tensor): Mask to prevent attention to future tokens.
        attn_dropout (nn.Dropout): Dropout layer for attention probabilities.
        output_proj (nn.Linear): Linear layer for projecting attention output.
        output_dropout (nn.Dropout): Dropout layer for output of the attention mechanism.
        ffn (nn.Sequential): Feed-forward network consisting of two linear layers with ReLU activation.
        ln1 (nn.LayerNorm): Layer normalization for attention output.
        ln2 (nn.LayerNorm): Layer normalization for feed-forward network output.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        
        head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer("attention_mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.attn_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.output_dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, input_tensor):
        """
        Forward pass of the self-attention and feed-forward network.
        
        Args:
            input_tensor (torch.Tensor): The input tensor with shape (batch_size, sequence_length, embed_dim).
        
        Returns:
            torch.Tensor: The output tensor after applying self-attention and feed-forward network.
        """
        B, T, C = input_tensor.shape
        
        keys = self.key_proj(input_tensor).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.query_proj(input_tensor).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_proj(input_tensor).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self.attention_mask[:, :, :T, :T]
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, values)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # Merge heads

        attn_output = self.output_dropout(self.output_proj(attn_output))
        attn_output = self.ln1(input_tensor + attn_output)

        ffn_output = self.ffn(attn_output)
        output_tensor = self.ln2(attn_output + ffn_output)

        return output_tensor

class GPTModel(nn.Module):
    """
    A simplified version of the GPT (Generative Pre-trained Transformer) model.
    
    Args:
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The dimension of the input embeddings.
        num_layers (int): The number of layers (self-attention and feed-forward blocks).
        num_heads (int): The number of attention heads.
        dropout (float): Dropout rate for regularization.
    
    Attributes:
        token_embeddings (nn.Embedding): Embedding layer for token representations.
        position_embeddings (nn.Embedding): Embedding layer for positional information.
        layers (nn.ModuleList): List of self-attention and feed-forward network layers.
        final_norm (nn.LayerNorm): Layer normalization for the final output.
        output_layer (nn.Linear): Linear layer for projecting hidden states to vocabulary logits.
    """

    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, dropout=0.1):
        super(GPTModel, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(block_size, embed_dim)
        self.layers = nn.ModuleList([SelfAttention(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the model.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, target_ids=None):
        """
        Forward pass of the GPT model.
        
        Args:
            input_ids (torch.Tensor): Input tensor with token ids, shape (batch_size, sequence_length).
            target_ids (torch.Tensor, optional): Target tensor with token ids, shape (batch_size, sequence_length).
        
        Returns:
            tuple: (logits, loss) where logits are the raw unnormalized predictions and loss is the cross-entropy loss if target_ids is provided.
        """
        B, T = input_ids.shape
        token_embeddings = self.token_embeddings(input_ids)
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        
        hidden_states = token_embeddings + position_embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        hidden_states = self.final_norm(hidden_states)
        logits = self.output_layer(hidden_states)

        loss = None
        if target_ids is not None:
            B, T, C = logits.size()
            logits = logits.view(B * T, C)
            target_ids = target_ids.view(B * T)
            loss = F.cross_entropy(logits, target_ids)
        
        return logits, loss

    def generate(self, input_ids, max_new_tokens):
        """
        Generate new tokens given a starting sequence.
        
        Args:
            input_ids (torch.Tensor): Input tensor with initial token ids, shape (batch_size, sequence_length).
            max_new_tokens (int): The maximum number of new tokens to generate.
        
        Returns:
            torch.Tensor: The tensor containing the generated sequence of token ids.
        """
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        
        return input_ids
