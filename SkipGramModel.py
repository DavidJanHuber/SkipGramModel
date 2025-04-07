import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import urllib.request
import zipfile

# Dataset Klasse - zum Datenset vorbereiten(Tensor etc.)
class SkipGramDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)

# die Modellklasse(ohne negative sampling)
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_words):
        center_vecs = self.in_embeddings(center_words)        # (B, D)
        output_vecs = self.out_embeddings.weight              # (V, D)
        scores = torch.matmul(center_vecs, output_vecs.T)     # (B, V)
        return scores

# Text8 laden
def load_text8(n_tokens=None):
    url = "http://mattmahoney.net/dc/text8.zip"
    urllib.request.urlretrieve(url, "text8.zip")
    with zipfile.ZipFile("text8.zip") as zf:
        text = zf.read("text8").decode("utf-8")
    tokens = text.split()
    return tokens[:n_tokens] if n_tokens else tokens

# Vokabular fertigmachen
def build_vocab(tokens, min_count=5):
    word_counts = Counter(w.lower() for w in tokens)
    filtered_words = [word for word, count in word_counts.items() if count >= min_count]

    vocab = {word: idx for idx, word in enumerate(filtered_words)}
    if "<unk>" not in vocab:
        vocab["<unk>"] = len(vocab)

    unk_idx = vocab["<unk>"]
    indexed = [vocab.get(word, unk_idx) for word in tokens]

    return vocab, indexed

# die Skip-Gram Paare generieren
def generate_pairs(indexed_tokens, window_size=2):
    result = []
    for i, center in enumerate(indexed_tokens):
        for j in range(max(i - window_size, 0), min(i + window_size + 1, len(indexed_tokens))):
            if i != j:
                result.append((center, indexed_tokens[j]))
    return result



# eigentliche Training
def train_model(pairs, vocab_size, embedding_dim=32, epochs=2, batch_size=128, lr=0.001):
    model = SkipGramModel(vocab_size, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dataset = SkipGramDataset(pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    i = 0
    for epoch in range(epochs):
        for centers, contexts in loader:
            optimizer.zero_grad()
            output = model(centers)
            loss = loss_fn(output, contexts)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f"Loss: {loss.item():.4f}")
            i += 1

    return model

# Wörter mit ähnlicher Bedeutung finden
def get_similar_words(word, vocab, embeddings, topn=5):
    word_idx = vocab.get(word, None)
    if word_idx is None:
        return f"'{word}' not in vocab"

    word_vec = embeddings[word_idx]
    sims = F.cosine_similarity(word_vec.unsqueeze(0), embeddings)

    top_indices = sims.topk(topn + 1).indices[1:]  # [1:] = ohne sich selbst
    idx_to_word = {idx: w for w, idx in vocab.items()}
    return [(idx_to_word[i.item()], sims[i].item()) for i in top_indices]


def main():
    tokens = load_text8(n_tokens=10000)
    vocab, indexed_tokens = build_vocab(tokens, min_count=5)
    pairs = generate_pairs(indexed_tokens, window_size=2)
    vocab_size = len(vocab)

    #Modell trainieren
    model = train_model(pairs, vocab_size=vocab_size)

    # Ähnliche Wörter finden
    embeddings = model.in_embeddings.weight.detach()
    similar = get_similar_words("king", vocab, embeddings)
    print("Ähnliche Wörter zu 'king':", similar)

    #Zusatzinfos
    print(f"Vokabulargröße: {vocab_size}")
    print(f"Anzahl Trainingspaare: {len(pairs)}")

    
    torch.save(model.state_dict(), "skipgram_model.pt")



if __name__ == "__main__":
    main()
