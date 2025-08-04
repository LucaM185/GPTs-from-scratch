import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

default_xkcd = 0.5
if "humor sans" in ([f.name.lower() for f in fm.fontManager.ttflist]):    
    plt.xkcd(default_xkcd)
    plt.rcParams['font.family'] = 'humor sans'
plt.rcParams['image.cmap'] = 'gray'

class xkcdoff:
    def __enter__(self):
        if "xkcd" in ([f.name for f in fm.fontManager.ttflist]):    
            plt.xkcd(0)
            plt.rcParams['font.family'] = 'humor sans'
        
    def __exit__(self, exc_type, exc_value, traceback):
        if "xkcd" in ([f.name for f in fm.fontManager.ttflist]):    
            plt.xkcd(default_xkcd)
            plt.rcParams['font.family'] = 'humor sans'

def linear(x, y):
    plt.xlabel('Number of hours of study')
    plt.ylabel('Test score')
    plt.title(f'{x.numel()} generated samples of students')

def logistic(x, y):
    # if values are only 1 and 0
    if y.unique().numel() == 2:
        plt.xlabel('Number of hours of study')
        plt.ylabel('Test passed (1) or failed (0)')
        plt.title(f'{x.numel()} generated samples of students')
    else:
        plt.xlabel('Number of hours of study')
        plt.ylabel('Probability of passing the test')
        plt.title(f'{x.numel()} generated samples of students')


def att_visualizations(activations):
    plt.imshow(activations[0].detach().cpu().numpy())
    plt.show()


import os
import torch
import matplotlib.pyplot as plt

att_idx = 0
def save_attention_heatmap(attn_weights, layer_idx, step=0, vis_dir="visualizations", save_ratio=0.01):
    global att_idx
    att_idx += 1
    if att_idx % int(1/save_ratio) == 0:
        os.makedirs(vis_dir, exist_ok=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(attn_weights.detach().cpu(), cmap='viridis')  # Avg over heads
        plt.title(f"Layer {layer_idx} Attention")
        plt.savefig(f"{vis_dir}/attn_layer{layer_idx}_step{step}.png")
        plt.close()

def save_activations(activations, name, vis_dir="visualizations"):
    os.makedirs(vis_dir, exist_ok=True)
    torch.save(activations, f"{vis_dir}/{name}.pt")

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DefaultAttention(nn.Module):
    """Default causal self-attention implementation"""
    def __init__(self, emb_size):
        super(DefaultAttention, self).__init__()
        self.emb_size = emb_size
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size) 

    def forward(self, x, **kwargs):
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)
        similarity = k @ q.transpose(-2, -1) / (self.emb_size**0.5)

        # Causal masking - current tokens can't see future tokens
        similarity[torch.tril(torch.ones_like(similarity)) == 0] = float("-inf")
        similarity = torch.softmax(similarity, dim=-1)

        attention = similarity @ v 
        return attention

class DefaultFullyConnected(nn.Module):
    """Default fully connected feedforward network"""
    def __init__(self, in_size, out_size, hidden_size, n_layers):
        super(DefaultFullyConnected, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fcx = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])
        self.fc2 = nn.Linear(hidden_size, out_size)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        for fc in self.fcx:
            x = F.gelu(fc(x))
        x = self.fc2(x)
        return x

class BaseBlock(nn.Module):
    """
    Configurable transformer block that accepts any attention and FC class
    """
    def __init__(self, emb_size, attention_class=None, fc_class=None, attention_kwargs=None, fc_kwargs=None):
        super().__init__()
        
        # Use defaults if not provided
        if attention_class is None:
            attention_class = DefaultAttention
        if fc_class is None:
            fc_class = DefaultFullyConnected
            
        self.prenorm1 = nn.LayerNorm(emb_size)
        
        # Initialize attention with provided class and kwargs
        attention_kwargs = attention_kwargs or {}
        self.att = attention_class(emb_size, **attention_kwargs)
        
        self.prenorm2 = nn.LayerNorm(emb_size)
        
        # Initialize fully connected with provided class and kwargs
        fc_kwargs = fc_kwargs or {}
        # Default FC kwargs if not provided
        if 'out_size' not in fc_kwargs:
            fc_kwargs['out_size'] = emb_size
        if 'hidden_size' not in fc_kwargs:
            fc_kwargs['hidden_size'] = emb_size * 4
        if 'n_layers' not in fc_kwargs:
            fc_kwargs['n_layers'] = 2
            
        self.fc = fc_class(emb_size, **fc_kwargs)

    def forward(self, x, **kwargs):
        x = x + self.att(self.prenorm1(x), **kwargs)
        x = x + self.fc(self.prenorm2(x))
        return x

class BaseTransformer(nn.Module):
    """
    Configurable transformer that accepts any attention and FC class
    
    Args:
        vocab_size: Size of vocabulary
        emb_size: Embedding dimension
        sequence_length: Maximum sequence length for positional embeddings
        n_blocks: Number of transformer blocks
        attention_class: Class to use for attention (defaults to DefaultAttention)
        fc_class: Class to use for feedforward (defaults to DefaultFullyConnected)
        attention_kwargs: Additional kwargs for attention class
        fc_kwargs: Additional kwargs for FC class
    """
    def __init__(self, vocab_size, emb_size, sequence_length, n_blocks, 
                 attention_class=None, fc_class=None, attention_kwargs=None, fc_kwargs=None):
        super().__init__()
        
        # Use defaults if not provided
        if attention_class is None:
            attention_class = DefaultAttention
        if fc_class is None:
            fc_class = DefaultFullyConnected
            
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.posemb = nn.Embedding(sequence_length, emb_size)
        
        # Create blocks with the specified attention and FC classes
        self.blocks = nn.ModuleList([
            BaseBlock(emb_size, attention_class, fc_class, attention_kwargs, fc_kwargs) 
            for _ in range(n_blocks)
        ])
        
        self.out = nn.Linear(emb_size, vocab_size)

    def forward(self, x, **kwargs):
        x = self.embedding(x) + self.posemb(torch.arange(x.size(1), device=x.device))
        
        # Handle autoregressive mode for cached attention
        if kwargs.get("autoregressive", False) and hasattr(self.blocks[-1].att, 'step') and self.blocks[-1].att.step != 0:
            x = x[:, -1:]
            
        for block in self.blocks:
            x = block(x, **kwargs)
            
        return self.out(x)
    
    def reset_cache(self):
        """Reset cache for attention mechanisms that support it"""
        for block in self.blocks:
            if hasattr(block.att, 'reset_cache'):
                block.att.reset_cache()
            elif hasattr(block.att, 'step'):
                block.att.step = 0

def train_model(model, inputs, labels, val_inputs, val_labels, vocab_size, 
                steps=2000, batch_size=1024, lr=3e-4, weight_decay=0.03, 
                device=None, print_every=None):
    """
    Reusable training function for any transformer variant
    
    Args:
        model: The transformer model to train
        inputs, labels: Training data (should support indexing and len())
        val_inputs, val_labels: Validation data
        vocab_size: Size of vocabulary
        steps: Number of training steps
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device to train on
        print_every: Print progress every N steps (defaults to steps//10)
    
    Returns:
        dict: Training history with 'train_loss' and 'val_loss' lists
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if print_every is None:
        print_every = max(1, steps // 10)
    
    model = model.to(device)
    try:
        model = torch.compile(model)
    except:
        print("Warning: torch.compile not available, using regular model")
    
    print(f"Millions of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    lossi = []
    vlossi = []
    
    for step in range(steps):
        # Training
        model.train()
        indexes = torch.randint(0, len(inputs), (batch_size,))
        
        pred = model(inputs[indexes].to(device))
        loss = F.cross_entropy(pred.view(-1, vocab_size), labels[indexes].to(device).view(-1))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        lossi.append(loss.item())
        
        # Validation
        with torch.no_grad():
            model.eval()
            val_batch_size = max(1, batch_size // 8)
            indexes = torch.randint(0, len(val_inputs), (val_batch_size,))
            pred = model(val_inputs[indexes].to(device))
            val_loss = F.cross_entropy(pred.view(-1, vocab_size), val_labels[indexes].to(device).view(-1))
            vlossi.append(val_loss.item())
        
        if step % print_every == 0:
            print(f"step {step:_>4d} - Loss: {lossi[-1]:.3f} - Val Loss: {vlossi[-1]:.3f}")
    
    return {'train_loss': lossi, 'val_loss': vlossi}

def plot_training_history(history, skip_initial=100):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 4))
    if len(history['train_loss']) > skip_initial:
        plt.plot(history['train_loss'][skip_initial:], label='Training Loss')
        plt.plot(history['val_loss'][skip_initial:], label='Validation Loss')
    else:
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.show()


def generate(model, string, sequence_length, device, texttoint, inttotext, num_tokens=300, **kwargs):
    """
    Generate text from a starting string
    
    Args:
        string: Starting string to generate from
        sequence_length: Maximum sequence length for model
        device: Device to run generation on
        texttoint: Dictionary mapping characters to integers
        inttotext: Dictionary mapping integers to characters
        num_tokens: Number of tokens to generate
        **kwargs: Additional kwargs to pass to forward
    """
    with torch.no_grad():
        for _ in range(num_tokens):
            X = torch.tensor([texttoint[s] for s in string[-sequence_length:]]).long().view(1, -1).to(device)
            pred = model.forward(X, **kwargs)
            string += inttotext[torch.multinomial(F.softmax(pred[0, -1, :], dim=0), 1).item()]
    return string

