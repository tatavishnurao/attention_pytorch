import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
from torch.utils.data import Dataset, DataLoader
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter
import io
import re
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# For demonstration, we'll use a small French-English dataset
# In a real scenario, you would use a larger dataset like WMT

# Sample data (small dataset for demonstration)
fr_en_data = [
    ("Je suis étudiant.", "I am a student."),
    ("Bonjour, comment ça va?", "Hello, how are you?"),
    ("J'aime la programmation.", "I love programming."),
    ("Où est la bibliothèque?", "Where is the library?"),
    ("Je vais au cinéma.", "I am going to the cinema."),
    ("Le chat est sur la table.", "The cat is on the table."),
    ("Je mange une pomme.", "I am eating an apple."),
    ("Il fait beau aujourd'hui.", "The weather is nice today."),
    ("Je ne parle pas français.", "I don't speak French."),
    ("Quelle heure est-il?", "What time is it?"),
    ("Je travaille à la maison.", "I work at home."),
    ("J'habite à Paris.", "I live in Paris."),
    ("Comment t'appelles-tu?", "What is your name?"),
    ("Je suis désolé.", "I am sorry."),
    ("Merci beaucoup.", "Thank you very much."),
    ("Au revoir!", "Goodbye!"),
    ("Bon appétit!", "Enjoy your meal!"),
    ("Bonne nuit.", "Good night."),
    ("À demain.", "See you tomorrow."),
    ("Je t'aime.", "I love you.")
]

# Create a custom dataset
class TranslationDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        
        # Tokenize and convert to indices
        src_tokens = self.src_tokenizer(src_text)
        tgt_tokens = self.tgt_tokenizer(tgt_text)
        
        # Add BOS and EOS tokens to target
        tgt_tokens = ['<bos>'] + tgt_tokens + ['<eos>']
        
        # Convert tokens to indices
        src_indices = [self.src_vocab[token] for token in src_tokens]
        tgt_indices = [self.tgt_vocab[token] for token in tgt_tokens]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

# Tokenizers
def get_fr_en_tokenizers():
    fr_tokenizer = get_tokenizer('spacy', language='fr')
    en_tokenizer = get_tokenizer('spacy', language='en')
    return fr_tokenizer, en_tokenizer

# Try to load spacy tokenizers, fallback to basic tokenizer if not available
try:
    fr_tokenizer, en_tokenizer = get_fr_en_tokenizers()
except:
    print("Spacy models not available. Using basic tokenizer.")
    # Basic tokenizer function
    def basic_tokenizer(text):
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    fr_tokenizer = basic_tokenizer
    en_tokenizer = basic_tokenizer

# Build vocabularies
def build_vocab(tokenizer, texts, special_tokens=None):
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    
    vocab = {'<pad>': 0, '<unk>': 1}
    if special_tokens:
        for token in special_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    
    return vocab

# Build vocabularies
fr_texts = [pair[0] for pair in fr_en_data]
en_texts = [pair[1] for pair in fr_en_data]

fr_vocab = build_vocab(fr_tokenizer, fr_texts)
en_vocab = build_vocab(en_tokenizer, en_texts, special_tokens=['<bos>', '<eos>'])

# Create reverse vocabularies for decoding
idx_to_fr = {idx: token for token, idx in fr_vocab.items()}
idx_to_en = {idx: token for token, idx in en_vocab.items()}

# Create dataset and dataloader
dataset = TranslationDataset(fr_en_data, fr_tokenizer, en_tokenizer, fr_vocab, en_vocab)

# Function to pad sequences in a batch
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    
    # Pad source sequences
    src_lengths = [len(seq) for seq in src_batch]
    max_src_len = max(src_lengths)
    padded_src = torch.zeros(len(batch), max_src_len, dtype=torch.long)
    for i, seq in enumerate(src_batch):
        padded_src[i, :len(seq)] = seq
    
    # Pad target sequences
    tgt_lengths = [len(seq) for seq in tgt_batch]
    max_tgt_len = max(tgt_lengths)
    padded_tgt = torch.zeros(len(batch), max_tgt_len, dtype=torch.long)
    for i, seq in enumerate(tgt_batch):
        padded_tgt[i, :len(seq)] = seq
    
    # Create source padding mask (1 for non-pad, 0 for pad)
    src_mask = (padded_src != 0).unsqueeze(1).unsqueeze(2)
    
    return padded_src, padded_tgt, src_mask

# Create dataloader
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def split_heads(self, x):
        # Reshape to separate heads
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_k)
    
    def combine_heads(self, x):
        # Combine heads back
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, d_k)
        return x.contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        # Linear projections and split heads
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        
        # Apply scaled dot-product attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply final linear projection
        output = self.W_o(self.combine_heads(attn_output))
        
        return output, attn_weights

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Self attention with residual connection and layer norm
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Self attention with residual connection and layer norm
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross attention with residual connection and layer norm
        cross_attn_output, cross_attn_weights = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attn_weights, cross_attn_weights

# Encoder
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, src, src_mask):
        # Embed tokens and add positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Store attention weights for visualization
        attn_weights = []
        
        # Pass through encoder layers
        for layer in self.layers:
            x, weights = layer(x, src_mask)
            attn_weights.append(weights)
        
        return x, attn_weights

# Decoder
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        # Embed tokens and add positional encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Store attention weights for visualization
        self_attn_weights = []
        cross_attn_weights = []
        
        # Pass through decoder layers
        for layer in self.layers:
            x, self_weights, cross_weights = layer(x, enc_output, src_mask, tgt_mask)
            self_attn_weights.append(self_weights)
            cross_attn_weights.append(cross_weights)
        
        # Final linear projection
        output = self.fc_out(x)
        
        return output, self_attn_weights, cross_attn_weights

# Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, 
                 num_heads=8, d_ff=2048, max_seq_length=100, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        # Pass through encoder
        enc_output, enc_attn_weights = self.encoder(src, src_mask)
        
        # Pass through decoder
        output, dec_self_attn_weights, dec_cross_attn_weights = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        return output, enc_attn_weights, dec_self_attn_weights, dec_cross_attn_weights
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)

# Create square subsequent mask for decoder
def create_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Initialize model
src_vocab_size = len(fr_vocab)
tgt_vocab_size = len(en_vocab)
d_model = 128  # Reduced for demonstration
num_layers = 3  # Reduced for demonstration
num_heads = 8
d_ff = 512  # Reduced for demonstration
max_seq_length = 50
dropout = 0.1

model = Transformer(
    src_vocab_size, 
    tgt_vocab_size, 
    d_model, 
    num_layers, 
    num_heads, 
    d_ff, 
    max_seq_length, 
    dropout
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Learning rate scheduler
class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        return self.d_model ** (-0.5) * min(self.current_step ** (-0.5), 
                                           self.current_step * self.warmup_steps ** (-1.5))

scheduler = WarmupScheduler(optimizer, d_model)

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt, src_mask) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)
        
        # Create target input and output
        tgt_input = tgt[:, :-1]  # Remove last token (EOS)
        tgt_output = tgt[:, 1:]  # Remove first token (BOS)
        
        # Create target mask
        tgt_mask = create_subsequent_mask(tgt_input.size(1)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output, _, _, _ = model(src, tgt_input, src_mask, tgt_mask)
        
        # Reshape for loss calculation
        output = output.contiguous().view(-1, output.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)
        
        # Calculate loss
        loss = criterion(output, tgt_output)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt, src_mask in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = src_mask.to(device)
            
            # Create target input and output
            tgt_input = tgt[:, :-1]  # Remove last token (EOS)
            tgt_output = tgt[:, 1:]  # Remove first token (BOS)
            
            # Create target mask
            tgt_mask = create_subsequent_mask(tgt_input.size(1)).to(device)
            
            # Forward pass
            output, _, _, _ = model(src, tgt_input, src_mask, tgt_mask)
            
            # Reshape for loss calculation
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Translation function
def translate(model, src_sentence, fr_tokenizer, fr_vocab, idx_to_en, device, max_len=50):
    model.eval()
    
    # Tokenize and convert to indices
    src_tokens = fr_tokenizer(src_sentence)
    src_indices = [fr_vocab.get(token, fr_vocab['<unk>']) for token in src_tokens]
    
    # Convert to tensor and add batch dimension
    src = torch.tensor([src_indices]).to(device)
    
    # Create source mask
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
    
    # Encode source
    enc_output, enc_attn_weights = model.encode(src, src_mask)
    
    # Initialize target with BOS token
    tgt = torch.tensor([[en_vocab['<bos>']]]).to(device)
    
    # Store attention weights for visualization
    all_dec_self_attn = []
    all_dec_cross_attn = []
    
    # Generate translation token by token
    for i in range(max_len):
        # Create target mask
        tgt_mask = create_subsequent_mask(tgt.size(1)).to(device)
        
        # Decode
        output, dec_self_attn, dec_cross_attn = model.decode(tgt, enc_output, src_mask, tgt_mask)
        
        # Get next token
        next_token = output[:, -1, :].argmax(dim=1).unsqueeze(1)
        
        # Append to target sequence
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Store attention weights
        all_dec_self_attn.append(dec_self_attn[-1][:, -1, :].detach().cpu())
        all_dec_cross_attn.append(dec_cross_attn[-1][:, -1, :].detach().cpu())
        
        # Break if EOS token is generated
        if next_token.item() == en_vocab['<eos>']:
            break
    
    # Convert indices to tokens
    tgt_indices = tgt[0, 1:-1].tolist()  # Remove BOS and EOS
    tgt_tokens = [idx_to_en[idx] for idx in tgt_indices]
    
    return ' '.join(tgt_tokens), src_tokens, all_dec_self_attn, all_dec_cross_attn, enc_attn_weights

# Visualize attention weights
def visualize_attention(src_tokens, tgt_tokens, attention_weights, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=src_tokens, yticklabels=tgt_tokens, cmap='viridis')
    plt.title(title)
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.tight_layout()
    plt.show()

# Train the model
num_epochs = 100
train_losses = []
eval_losses = []

print("Starting training...")
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss = train_epoch(model, dataloader, optimizer, scheduler, criterion, device)
    train_losses.append(train_loss)
    
    # Evaluate every 5 epochs
    if (epoch + 1) % 5 == 0:
        eval_loss = evaluate(model, dataloader, criterion, device)
        eval_losses.append(eval_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f} | Time: {time.time() - start_time:.2f}s")
    else:
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Time: {time.time() - start_time:.2f}s")

# Plot training and evaluation losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(5, num_epochs + 1, 5), eval_losses, label='Eval Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Losses')
plt.legend()
plt.grid(True)
plt.show()

# Test translation with visualization
test_sentences = [
    "Je suis étudiant.",
    "Bonjour, comment ça va?",
    "J'aime la programmation.",
    "Où est la bibliothèque?"
]

for test_sentence in test_sentences:
    translation, src_tokens, dec_self_attn, dec_cross_attn, enc_attn = translate(
        model, test_sentence, fr_tokenizer, fr_vocab, idx_to_en, device
    )
    
    print(f"Source: {test_sentence}")
    print(f"Translation: {translation}")
    print()
    
    # Visualize encoder self-attention (last layer, first head)
    enc_attn_last_layer = enc_attn[-1][0, 0].detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(enc_attn_last_layer, xticklabels=src_tokens, yticklabels=src_tokens, cmap='viridis')
    plt.title(f"Encoder Self-Attention\nSource: {test_sentence}")
    plt.xlabel('Source Tokens')
    plt.ylabel('Source Tokens')
    plt.tight_layout()
    plt.show()
    
    # Visualize decoder cross-attention (last token)
    if len(dec_cross_attn) > 0:
        # Get the cross-attention for the last generated token
        cross_attn_last_token = dec_cross_attn[-1][0, 0].detach().cpu().numpy()
        
        # Get the translated tokens (excluding BOS and EOS)
        translated_tokens = translation.split()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(cross_attn_last_token.reshape(1, -1), 
                   xticklabels=src_tokens, 
                   yticklabels=[translated_tokens[-1]], 
                   cmap='viridis')
        plt.title(f"Decoder Cross-Attention for Last Token\nSource: {test_sentence}\nTranslation: {translation}")
        plt.xlabel('Source Tokens')
        plt.ylabel('Target Token')
        plt.tight_layout()
        plt.show()