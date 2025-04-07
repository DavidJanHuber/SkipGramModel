import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
import urllib.request
import zipfile

# Dataset-Klasse für Negative Sampling
class SkipGramNSDataset(Dataset):
    def __init__(self, pairs, vocab, k_neg=5):
        self.pairs = pairs
        self.vocab = vocab
        self.k = k_neg
        self.word_indices = list(vocab.values())

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        negatives = []

        while len(negatives) < self.k:
            neg = random.choice(self.word_indices)
            if neg != context and neg != center:
                negatives.append(neg)

        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long)
        )

# die Modell-Klasse für SGNS
class SkipGramNSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, context, negatives):
        center_vecs = self.in_embeddings(center)                  # (B, D)
        context_vecs = self.out_embeddings(context)               # (B, D)
        negative_vecs = self.out_embeddings(negatives)            # (B, K, D)

        pos_score = torch.sum(center_vecs * context_vecs, dim=1)  # (B,)
        neg_score = torch.bmm(negative_vecs, center_vecs.unsqueeze(2)).squeeze()  # (B, K)

        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1)

        return (pos_loss + neg_loss).mean()

# Funktion: Daten laden
def load_text8(n_tokens=None):
    url = "http://mattmahoney.net/dc/text8.zip"
    urllib.request.urlretrieve(url, "text8.zip")
    with zipfile.ZipFile("text8.zip") as zf:
        text = zf.read("text8").decode("utf-8")
    tokens = text.split()
    return tokens[:n_tokens] if n_tokens else tokens

# Vokabular fertig machen
def build_vocab(tokens, min_count=5):
    word_counts = Counter(w.lower() for w in tokens)
    filtered_words = [word for word, count in word_counts.items() if count >= min_count]

    vocab = {word: idx for idx, word in enumerate(filtered_words)}
    if "<unk>" not in vocab:
        vocab["<unk>"] = len(vocab)

    unk_idx = vocab["<unk>"]
    indexed_tokens = [vocab.get(word, unk_idx) for word in tokens]
    return vocab, indexed_tokens

# die Skip-Gram Paare generieren
def generate_pairs(indexed_tokens, window_size=2):
    pairs = []
    for i in range(window_size, len(indexed_tokens) - window_size):
        center = indexed_tokens[i]
        context = indexed_tokens[i - window_size:i] + indexed_tokens[i + 1:i + window_size + 1]
        for ctx in context:
            pairs.append((center, ctx))
    return pairs

# eigentliche Trainingsfunktion
def train_model(pairs, vocab, embedding_dim=32, batch_size=128, epochs=3, k_neg=5, lr=0.003, device="cpu"):
    dataset = SkipGramNSDataset(pairs, vocab, k_neg=k_neg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SkipGramNSModel(vocab_size=len(vocab), embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    step = 0
    for epoch in range(epochs):
        for center, context, negatives in loader:
            center = center.to(device)
            context = context.to(device)
            negatives = negatives.to(device)

            loss = model(center, context, negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
            step += 1

    return model

# Testfunktion: Ähnliche Wörter finden
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
    # Hyperparameter
    embedding_dim = 32
    batch_size = 128
    epochs = 3
    k_neg = 5
    lr = 0.003
    context_window = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Daten vorbereiten
    tokens = load_text8(n_tokens=10000)
    vocab, indexed_tokens = build_vocab(tokens, min_count=5)
    pairs = generate_pairs(indexed_tokens, window_size=context_window)

    print(f"Vokabulargröße: {len(vocab)}")
    print(f"Trainingspaare: {len(pairs)}")

    # Modell trainieren
    model = train_model(pairs, vocab, embedding_dim, batch_size, epochs, k_neg, lr, device)

    # Speichern
    torch.save(model.state_dict(), "skipgram_negativesampling_model.pt")
    print("Modell gespeichert unter 'skipgram_negativesampling_model.pt'")
    embeddings = model.in_embeddings.weight.detach()
    similar = get_similar_words("king", vocab, embeddings)
    print("Ähnliche Wörter zu 'king':", similar)


if __name__ == "__main__":
    main()
