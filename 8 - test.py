import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import LucaM185 
from torch.utils.data import DataLoader
device = torch.device("cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

with open("datasets/enwik8", "r") as f:
    load = f.read()[:10000]
print(load[:200])
class MyInputs:
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx): # idx is an int
        return self.data[idx:idx+self.sequence_length]
# Restricting to ASCII characters
texttoint = {chr(i): i for i in range(256)}
inttotext = {i: chr(i) for i in range(256)}
dataset = torch.tensor([texttoint[c] for c in load if c in texttoint])

vocab_size = len(texttoint)
sequence_length = 20
val_set = int(len(dataset)*0.1)

inputs = MyInputs(dataset[val_set:-1-sequence_length], sequence_length)
labels = MyInputs(dataset[val_set+1:-sequence_length], sequence_length)
val_inputs = MyInputs(dataset[:val_set], sequence_length)
val_labels = MyInputs(dataset[1:val_set+1], sequence_length)

print(len(dataset))
import math

class FullyConnected(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, n_layers, cache_len, batch_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fcx = nn.Sequential(*[nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])
        self.fc2 = nn.Linear(hidden_size, out_size)
        self.register_buffer("cache", torch.zeros(batch_size, cache_len, out_size))

    def forward(self, x):
        if self.training:
            return self.fc2(self.fcx(self.fc1(x)))
        else:
            self.cache[:, :x.shape[1], :] = self.fc2(self.fcx(self.fc1(x)))
            return self.cache[:, :x.shape[1], :]

class KVAttention(nn.Module):
    """Fixed-size KV cache. Refill at training-time, append at eval-time."""
    def __init__(self, emb_size: int, cache_len, batch_size):
        super().__init__()
        self.emb_size  = emb_size
        self.cache_len = cache_len

        self.keys    = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values  = nn.Linear(emb_size, emb_size)

        self.register_buffer("cache_k",  torch.zeros(batch_size, cache_len, emb_size))
        self.register_buffer("cache_v",  torch.zeros(batch_size, cache_len, emb_size))
        self.register_buffer("cache_qk", torch.zeros(batch_size, cache_len, cache_len))
        self.step = 0        # next free slot

    def reset_cache(self):                      # call this before a new decode
        self.step = 0
        # Clear all cache tensors
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.cache_qk.zero_()

    def get_causal_mask(self, seq_len: int):
        """Create a causal mask for the given sequence length."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.cache_k.device), diagonal=1)
        return mask.bool()

    def forward(self, x, use_cache = None, reset_cache = False):
        """
        use_cache=None  ➜  default: training → False, eval → True
        reset_cache=True   clears the cache at the start of a generation pass
        """
        if use_cache is None:
            use_cache = not self.training
        if reset_cache:
            self.reset_cache()

        # During training, always do full computation without cache
        if self.training or not use_cache or self.step == 0:
            qk, v = self.compute_qkv(x)
        else:
            qk, v = self.update_qkv(x)

        # Apply attention
        att = torch.softmax(qk / math.sqrt(self.emb_size), dim=-1)
        result = torch.matmul(att, v)
        
        # Ensure result matches input batch size
        if result.shape[0] != x.shape[0]:
            result = result[:x.shape[0], :, :]
        if result.shape[1] != x.shape[1]:
            result = result[:, :x.shape[1], :]
            
        return result

    def compute_qkv(self, x):
        L = x.shape[1]
        B = x.shape[0]

        q, k, v = self.queries(x), self.keys(x), self.values(x)
        qk = torch.matmul(q, k.transpose(-2, -1))        # (B, L, L)

        # Apply causal mask
        mask = self.get_causal_mask(L)
        qk = qk.masked_fill(mask, -1e9)

        # Only cache during eval mode
        if not self.training:
            # Handle cache length limit
            if L > self.cache_len:
                L = self.cache_len
                q, k, v = q[:, :L, :], k[:, :L, :], v[:, :L, :]
                qk = qk[:, :L, :L]

            # Only cache up to the actual batch size being used
            cache_batch_size = min(B, self.cache_k.shape[0])
            self.cache_k[:cache_batch_size, :L, :] = k[:cache_batch_size, :, :]
            self.cache_v[:cache_batch_size, :L, :] = v[:cache_batch_size, :, :]
            self.cache_qk[:cache_batch_size, :L, :L] = qk[:cache_batch_size, :, :]
            self.step = L
            
        return qk, v

    def update_qkv(self, x):
        if self.step == 0: 
            return self.compute_qkv(x)
        
        B = x.shape[0]
        cache_batch_size = min(B, self.cache_k.shape[0])
        
        q = self.queries(x[:, -1, :]).unsqueeze(1)       # (B, 1, E)
        k = self.keys(x[:, -1, :]).unsqueeze(1)          # (B, 1, E)
        v = self.values(x[:, -1, :]).unsqueeze(1)        # (B, 1, E)

        # write k and v only for the actual batch size being used
        self.cache_k[:cache_batch_size, self.step:self.step+1, :] = k[:cache_batch_size, :, :]
        self.cache_v[:cache_batch_size, self.step:self.step+1, :] = v[:cache_batch_size, :, :]

        # q against all keys so far (incl. the new one)
        qk_row = torch.matmul(q, self.cache_k[:cache_batch_size, :self.step+1, :].transpose(-2, -1))
        self.cache_qk[:cache_batch_size, self.step:self.step+1, :self.step+1] = qk_row

        self.step += 1
        
        # Apply causal mask to the cached QK matrix
        current_qk = self.cache_qk[:cache_batch_size, :self.step, :self.step]
        mask = self.get_causal_mask(self.step)
        current_qk = current_qk.masked_fill(mask, -1e9)
        
        # Return only the relevant portion of cache for the current batch size
        return (current_qk, self.cache_v[:cache_batch_size, :self.step, :])

class Attention(nn.Module):
    def __init__(self, emb_size, cache_len, batch_size):
        super().__init__()
        self.kv = KVAttention(emb_size, cache_len, batch_size)

    def forward(self, x, **kw):
        return self.kv(x, **kw)

class Block(nn.Module):
    def __init__(self, emb_size, hidden_size, n_layers, cache_len, batch_size):
        super().__init__()
        self.prenorm1 = nn.LayerNorm(emb_size)
        self.att      = Attention(emb_size, cache_len, batch_size)
        self.prenorm2 = nn.LayerNorm(emb_size)
        self.fc       = FullyConnected(emb_size, emb_size, hidden_size, n_layers, cache_len, batch_size)

    def forward(self, x, **kw):
        x = x + self.att(self.prenorm1(x), **kw)
        x = x + self.fc(self.prenorm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_size, n_blocks,
                 head_size, hidden_size, n_layers, sequence_length, cache_len, batch_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.posemb    = nn.Embedding(sequence_length, emb_size)
        self.blocks    = nn.ModuleList(
            [Block(emb_size, hidden_size, n_layers, cache_len, batch_size) for _ in range(n_blocks)]
        )
        self.out = nn.Linear(emb_size, vocab_size)
        self.sequence_length = sequence_length

    # propagate use_cache / reset_cache downwards
    def forward(self, x, **kw):
        # Ensure position embeddings don't exceed sequence_length
        pos_ids = torch.arange(x.size(1), device=x.device)
        if x.size(1) > self.sequence_length:
            pos_ids = pos_ids % self.sequence_length
        
        x = self.embedding(x) + self.posemb(pos_ids)
        for blk in self.blocks:
            x = blk(x, **kw)
        return self.out(x)

    # convenience helper
    def reset_all_caches(self):
        for m in self.modules():
            if isinstance(m, KVAttention):
                m.reset_cache()

# Fix model instantiation to use correct vocab_size and batch_size for training
model = Transformer(vocab_size=vocab_size, emb_size=64, n_blocks=2, head_size=64, 
                   hidden_size=64, n_layers=1, sequence_length=sequence_length, 
                   cache_len=1024, batch_size=8).to(device)

def create_batch_dataset(inputs, labels, batch_size):
    """Create a simple dataset that yields batches of inputs and labels"""
    class BatchDataset:
        def __init__(self, inputs, labels):
            # Convert MyInputs objects to lists for easier handling
            self.inputs = [inputs[i] for i in range(len(inputs))]
            self.labels = [labels[i] for i in range(len(labels))]
            
        def __len__(self):
            return len(self.inputs)
            
        def __getitem__(self, idx):
            input_item = self.inputs[idx]
            label_item = self.labels[idx]
            # Convert to tensor if not already a tensor
            if not isinstance(input_item, torch.Tensor):
                input_item = torch.tensor(input_item, dtype=torch.long)
            else:
                input_item = input_item.clone().detach().long()
                
            if not isinstance(label_item, torch.Tensor):
                label_item = torch.tensor(label_item, dtype=torch.long)
            else:
                label_item = label_item.clone().detach().long()
                
            return input_item, label_item
    
    return BatchDataset(inputs, labels)

# Fast training function for testing
def train_model_fast(model, train_inputs, train_labels, val_inputs, val_labels, 
                    epochs=2, batch_size=8, learning_rate=1e-3):
    
    # Limit training data for speed
    train_limit = min(1000, len(train_inputs))
    val_limit = min(200, len(val_inputs))
    
    # Create limited datasets by manually extracting samples
    train_input_list = [train_inputs[i] for i in range(train_limit)]
    train_label_list = [train_labels[i] for i in range(train_limit)]
    val_input_list = [val_inputs[i] for i in range(val_limit)]
    val_label_list = [val_labels[i] for i in range(val_limit)]
    
    # Create simple dataset classes
    class SimpleDataset:
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels
            
        def __len__(self):
            return len(self.inputs)
            
        def __getitem__(self, idx):
            input_item = self.inputs[idx]
            label_item = self.labels[idx]
            # Convert to tensor if not already a tensor
            if not isinstance(input_item, torch.Tensor):
                input_item = torch.tensor(input_item, dtype=torch.long)
            else:
                input_item = input_item.clone().detach().long()
                
            if not isinstance(label_item, torch.Tensor):
                label_item = torch.tensor(label_item, dtype=torch.long)
            else:
                label_item = label_item.clone().detach().long()
                
            return input_item, label_item
    
    train_dataset = SimpleDataset(train_input_list, train_label_list)
    val_dataset = SimpleDataset(val_input_list, val_label_list)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    print(f"Fast training: {epochs} epochs, {train_limit} samples, batch_size={batch_size}")
    
    for epoch in range(epochs):
        total_loss = 0
        total_batches = 0
        
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - cache is disabled during training automatically
            outputs = model(batch_inputs)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, vocab_size), batch_labels.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
        
        # Quick validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_inputs)
                loss = criterion(outputs.view(-1, vocab_size), batch_labels.view(-1))
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = total_loss / total_batches if total_batches > 0 else 0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        model.train()

# Training function
def train_model(model, train_inputs, train_labels, val_inputs, val_labels, 
                epochs=10, batch_size=32, learning_rate=3e-4):
    
    # Create datasets
    train_dataset = create_batch_dataset(train_inputs, train_labels, batch_size)
    val_dataset = create_batch_dataset(val_inputs, val_labels, batch_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_batches = 0
        
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - cache is disabled during training automatically
            outputs = model(batch_inputs)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, vocab_size), batch_labels.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
            if total_batches % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {total_batches}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_inputs)
                loss = criterion(outputs.view(-1, vocab_size), batch_labels.view(-1))
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = total_loss / total_batches
        avg_val_loss = val_loss / val_batches
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        model.train()

# Inference function for text generation WITH cache (proper incremental usage)
def generate_text_with_cache(model, prompt_text, max_length=50, temperature=1.0):
    model.eval()
    model.reset_all_caches()
    
    # Convert prompt to tokens
    prompt_tokens = [texttoint[c] for c in prompt_text if c in texttoint]
    if not prompt_tokens:
        prompt_tokens = [texttoint[' ']]  # fallback to space
    
    generated_tokens = prompt_tokens.copy()
    
    with torch.no_grad():
        # First, process the entire prompt to populate cache
        input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        outputs = model(input_tensor, use_cache=True, reset_cache=True)
        
        # Then generate tokens one by one using incremental cache updates
        for _ in range(max_length - len(prompt_tokens)):
            # Get probabilities for the last token
            logits = outputs[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1).item()
            generated_tokens.append(next_token)
            
            # Process just the new token using cache
            next_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            outputs = model(next_input, use_cache=True, reset_cache=False)
    
    # Convert back to text
    generated_text = ''.join([inttotext.get(token, '?') for token in generated_tokens])
    return generated_text

# Inference function for text generation WITHOUT cache (for comparison)
def generate_text_without_cache(model, prompt_text, max_length=50, temperature=1.0):
    model.eval()
    
    # Convert prompt to tokens
    prompt_tokens = [texttoint[c] for c in prompt_text if c in texttoint]
    if not prompt_tokens:
        prompt_tokens = [texttoint[' ']]  # fallback to space
    
    generated_tokens = prompt_tokens.copy()
    
    with torch.no_grad():
        # Generate tokens one by one WITHOUT using cache
        for _ in range(max_length - len(prompt_tokens)):
            # Always process the full sequence so far
            input_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            
            # Forward pass without cache
            outputs = model(input_tensor, use_cache=False, reset_cache=False)
            
            # Get probabilities for the last token
            logits = outputs[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1).item()
            generated_tokens.append(next_token)
    
    # Convert back to text
    generated_text = ''.join([inttotext.get(token, '?') for token in generated_tokens])
    return generated_text

import time

def compare_generation_speed(model, prompt, max_length=50, temperature=0.8):
    """Compare generation speed with and without cache"""
    print(f"\n=== Generation Speed Comparison ===")
    print(f"Prompt: '{prompt}'")
    print(f"Generating {max_length} tokens...")
    
    # Ensure clean state before each test
    model.reset_all_caches()
    
    # Set same random seed for deterministic comparison
    torch.manual_seed(42)
    start_time = time.time()
    text_with_cache = generate_text_with_cache(model, prompt, max_length, temperature)
    time_with_cache = time.time() - start_time
    
    # Ensure clean state before second test
    model.reset_all_caches()
    
    # Set same random seed for deterministic comparison
    torch.manual_seed(42)
    start_time = time.time()
    text_without_cache = generate_text_without_cache(model, prompt, max_length, temperature)
    time_without_cache = time.time() - start_time
    
    print(f"\nWITH Cache:")
    print(f"Time: {time_with_cache:.3f}s")
    print(f"Text: '{text_with_cache}'")
    
    print(f"\nWITHOUT Cache:")
    print(f"Time: {time_without_cache:.3f}s") 
    print(f"Text: '{text_without_cache}'")
    
    # Check if results are identical
    if text_with_cache == text_without_cache:
        print("✅ Results are identical!")
    else:
        print("❌ Results differ - cache bug detected!")
    
    speedup = time_without_cache / time_with_cache if time_with_cache > 0 else float('inf')
    print(f"\nSpeedup with cache: {speedup:.2f}x")
    
    return time_with_cache, time_without_cache

# Inference function for text generation
def generate_text(model, prompt_text, max_length=100, temperature=1.0):
    model.eval()
    model.reset_all_caches()
    
    # Convert prompt to tokens
    prompt_tokens = [texttoint[c] for c in prompt_text if c in texttoint]
    if not prompt_tokens:
        prompt_tokens = [texttoint[' ']]  # fallback to space
    
    generated_tokens = prompt_tokens.copy()
    
    with torch.no_grad():
        # Process the prompt
        input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        # First forward pass with reset_cache=True
        outputs = model(input_tensor, use_cache=True, reset_cache=True)
        
        # Generate tokens one by one
        for _ in range(max_length - len(prompt_tokens)):
            # Get probabilities for the last token
            logits = outputs[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1).item()
            generated_tokens.append(next_token)
            
            # Prepare input for next iteration (just the new token)
            next_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            
            # Forward pass using cache
            outputs = model(next_input, use_cache=True, reset_cache=False)
    
    # Convert back to text
    generated_text = ''.join([inttotext.get(token, '?') for token in generated_tokens])
    return generated_text

def debug_cache_equivalence():
    """Debug function to test if cache computation is equivalent to non-cache"""
    model.eval()
    
    # Simple test sequence
    test_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)
    
    print("=== Cache Equivalence Test ===")
    
    # Method 1: Full computation (no cache)
    with torch.no_grad():
        output_no_cache = model(test_input, use_cache=False)
    
    # Method 2: With cache 
    model.reset_all_caches()
    with torch.no_grad():
        output_with_cache = model(test_input, use_cache=True, reset_cache=True)
    
    print(f"No cache output shape: {output_no_cache.shape}")
    print(f"With cache output shape: {output_with_cache.shape}")
    print(f"Outputs are close: {torch.allclose(output_no_cache, output_with_cache, atol=1e-6)}")
    
    if not torch.allclose(output_no_cache, output_with_cache, atol=1e-6):
        print(f"Max difference: {torch.max(torch.abs(output_no_cache - output_with_cache))}")
        print("❌ Cache computation is not equivalent!")
        return False
    else:
        print("✅ Cache computation is equivalent!")
        return True

# Example usage
if __name__ == "__main__":
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocab size: {vocab_size}")
    print(f"Training samples: {len(inputs)}")
    print(f"Validation samples: {len(val_inputs)}")
    
    # Fast training for testing
    print("\n=== Starting Fast Training ===")
    start_train_time = time.time()
    train_model_fast(model, inputs, labels, val_inputs, val_labels, epochs=2, batch_size=8)
    train_time = time.time() - start_train_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # First test cache equivalence
    debug_cache_equivalence()
    
    # Test generation with different speeds
    prompts = ["The quick", "Hello", "In the"]
    
    for prompt in prompts:
        compare_generation_speed(model, prompt, max_length=30, temperature=0.8)



