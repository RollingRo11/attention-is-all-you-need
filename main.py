import torch
import torch.nn as nn
import sys
from torch.nn import functional as F
from model import BigramLanguageModel
from data import load_data, get_batch

# hyperparameters
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 200

torch.manual_seed(1337)

# Load data
train_data, val_data, vocab_size, encode, decode = load_data()


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Initialize model
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Check if we're doing inference only
if len(sys.argv) > 1 and sys.argv[1] == "inference":
    print("Loading pretrained model for inference...")
    m.load_state_dict(torch.load("model.pt", map_location=device))
    m.eval()

    # Optionally, allow user to provide a prompt
    prompt = "ROMEO:"
    if len(sys.argv) > 2:
        prompt = sys.argv[2]
    if prompt:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        generated_indices = m.generate(context, max_new_tokens=500)[0].tolist()
    print("Generated indices:", generated_indices)  # For debugging
    generated_text = decode(generated_indices)
    print("Generated text:")
    print(generated_text)
    sys.exit(0)

# Training
print("Training model...")
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample 1 batch
    xb, yb = get_batch(train_data, val_data, "train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save
torch.save(m.state_dict(), "model.pt")
print("Model saved to model.pt")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
