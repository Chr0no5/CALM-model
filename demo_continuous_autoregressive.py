import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_corpus():
    samples = [
        "the cat sat on the mat",
        "the dog barked at the moon",
        "language models learn by predicting tokens",
        "continuous autoregressive generation is a fascinating idea",
        "small proofs of concept help us understand large systems",
    ]
    return "\n".join(samples)


def causal_mask(seq_len, device):
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    return mask


class ContinuousARTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, embed_dim=64, n_heads=4, hidden_dim=128, n_layers=2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.to_logits = nn.Linear(embed_dim, vocab_size)
        self.to_latent = nn.Linear(embed_dim, embed_dim)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embed(x) + self.pos_embed(positions)
        mask = causal_mask(seq_len, x.device)
        hidden = self.transformer(x, mask=mask)
        logits = self.to_logits(hidden)
        latent = self.to_latent(hidden)
        return logits, latent


def decode_continuous_embedding(predicted_embedding, embedding_weights):
    similarity = F.cosine_similarity(
        predicted_embedding.unsqueeze(0), embedding_weights, dim=-1
    )
    return torch.argmax(similarity, dim=-1).item()


def main():
    random.seed(7)
    torch.manual_seed(7)

    text = build_corpus()
    chars = sorted(set(text.lower()))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(chars)

    data = torch.tensor([stoi[c] for c in text.lower()], dtype=torch.long)
    model = ContinuousARTransformer(vocab_size=vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step in range(1000):
        start = random.randint(0, len(data) - 32)
        chunk = data[start : start + 32]
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)

        logits, latent = model(x)
        target_embeddings = model.token_embed(y)

        ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        latent_loss = F.mse_loss(latent, target_embeddings)
        loss = ce_loss + 0.5 * latent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(
                f"step={step:03d} ce={ce_loss.item():.4f} latent={latent_loss.item():.4f} total={loss.item():.4f}"
            )

    model.eval()
    seed = "lang"
    context = torch.tensor([stoi.get(c, 0) for c in seed.lower()], dtype=torch.long).unsqueeze(0)
    generated = list(seed)
    generated_continuous = list(seed)

    with torch.no_grad():
        for _ in range(40):
            logits, latent = model(context)
            last_logits = logits[:, -1, :]
            last_latent = latent[:, -1, :]

            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(itos[next_token])

            discrete_choice = decode_continuous_embedding(last_latent[0], model.token_embed.weight)
            generated_continuous.append(itos[discrete_choice])

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            context = torch.cat([context, next_token_tensor], dim=1)
            if context.size(1) > model.max_seq_len:
                context = context[:, -model.max_seq_len :]

    print("\nGenerated sample (standard autoregressive sampling):")
    print("".join(generated))
    print("\nGenerated sample (continuous embedding decode):")
    print("".join(generated_continuous))
    print(
        "\nThis model is trained with two goals: discrete next-token prediction and continuous next-embedding prediction."
    )
    print(
        "The second generation path uses the model's predicted embedding to choose a token, demonstrating a continuous autoregressive decoding step."
    )


if __name__ == "__main__":
    main()