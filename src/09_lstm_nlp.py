import torch
import torch.nn as nn
import torch.optim as optim

# 1. Setup Mock Data
vocab = {"<PAD>": 0, "great": 1, "movie": 2, "terrible": 3, "acting": 4, "loved": 5, "it": 6}
vocab_size = len(vocab)
embedding_dim = 8
hidden_dim = 16
output_dim = 1

X_train = torch.tensor([
    [5, 6, 1, 2],  # "loved it great movie"
    [3, 2, 3, 4]   # "terrible movie terrible acting"
], dtype=torch.long)

y_train = torch.tensor([1.0, 0.0]).unsqueeze(1)


# 2. Define LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Swapped nn.RNN for nn.LSTM!
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)
        
        # LSTM returns: output, (hidden_state, cell_state)
        # output: [batch_size, seq_len, hidden_dim]
        # hidden: [1, batch_size, hidden_dim]
        # cell:   [1, batch_size, hidden_dim]
        output, (hidden, cell) = self.lstm(embedded)
        
        # Extract the last hidden state
        last_hidden = hidden.squeeze(0)
        
        logits = self.fc(last_hidden)
        return logits


# 3. Train Model
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training LSTM Model...\n")
for epoch in range(1, 101):
    optimizer.zero_grad()
    
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 25 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# 4. Evaluation
with torch.no_grad():
    scores = torch.sigmoid(model(X_train))
    print("\nLSTM Predictions:")
    print(f"Sample 1 ('loved it great movie'): {scores[0].item():.4f} (Target: 1.0)")
    print(f"Sample 2 ('terrible movie terrible acting'): {scores[1].item():.4f} (Target: 0.0)")