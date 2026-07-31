import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------
# 1. Mock Data & Vocabulary Setup
# ---------------------------------------------------------------
# Simple vocabulary mapping words to unique integer IDs
vocab = {"<PAD>": 0, "great": 1, "movie": 2, "terrible": 3, "acting": 4, "loved": 5, "it": 6}
vocab_size = len(vocab)
embedding_dim = 8   # Vector size for each word
hidden_dim = 16     # Size of the RNN's internal memory
output_dim = 1      # Binary classification (0 or 1)

# Sample training batch (encoded as word IDs)
# "loved it great movie" -> Positive (1)
# "terrible movie terrible acting" -> Negative (0)
X_train = torch.tensor([
    [5, 6, 1, 2],  # Batch item 1
    [3, 2, 3, 4]   # Batch item 2
], dtype=torch.long)

y_train = torch.tensor([1.0, 0.0]).unsqueeze(1) # Labels shape: (batch_size, 1)


# ---------------------------------------------------------------
# 2. Define the RNN Architecture
# ---------------------------------------------------------------
class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleRNNClassifier, self).__init__()
        
        # Maps word IDs to dense continuous vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # The core RNN layer
        # batch_first=True expects input shape: (batch_size, sequence_length, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        
        # Linear layer to map the final hidden state to a single prediction score
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        
        embedded = self.embedding(text) 
        # embedded shape: [batch_size, seq_len, embed_dim]
        
        # output contains all hidden states across all time steps
        # hidden contains ONLY the final hidden state of the sequence
        output, hidden = self.rnn(embedded)
        
        # We take the final hidden state squeeze to shape [batch_size, hidden_dim]
        last_hidden = hidden.squeeze(0)
        
        # Pass through the linear layer to get the final score
        logits = self.fc(last_hidden)
        return logits


# ---------------------------------------------------------------
# 3. Model Training Loop
# ---------------------------------------------------------------
model = SimpleRNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.BCEWithLogitsLoss() # Combines Sigmoid layer + Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Starting training...\n")
for epoch in range(1, 101):
    optimizer.zero_grad()
    
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 25 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# ---------------------------------------------------------------
# 4. Evaluation / Quick Test
# ---------------------------------------------------------------
with torch.no_grad():
    test_scores = torch.sigmoid(model(X_train))
    print("\nPredictions after training:")
    print(f"Sample 1 ('loved it great movie'): {test_scores[0].item():.4f} (Target: 1.0)")
    print(f"Sample 2 ('terrible movie terrible acting'): {test_scores[1].item():.4f} (Target: 0.0)")