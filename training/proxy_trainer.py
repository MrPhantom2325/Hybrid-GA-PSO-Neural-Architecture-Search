
"""
training/proxy_trainer.py
Short proxy training to get a cheap fitness estimate for GA.
Used in Phase 3 (GA fitness). Will be replaced by surrogate in Phase 5.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim


def proxy_train(
    model,
    train_loader,
    val_loader,
    device,
    epochs        = 3,
    lr            = 1e-3,
    dropout_rate  = 0.3,
    weight_decay  = 1e-4,
    verbose       = False,
):
    """
    Train model for `epochs` epochs. Return dict with:
      - val_accuracy  : float  (primary fitness signal for GA)
      - num_params    : int
      - train_time    : float  seconds
      - train_losses  : list
      - val_accs      : list
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine annealing over proxy epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    num_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_losses = []
    val_accs     = []
    t0           = time.time()

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        train_losses.append(epoch_loss / len(train_loader))

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds   = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)
        val_acc = correct / total
        val_accs.append(val_acc)

        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}  "
                  f"loss={train_losses[-1]:.4f}  val_acc={val_acc:.4f}")

    train_time = time.time() - t0

    return {
        "val_accuracy" : val_accs[-1],      # last epoch accuracy
        "best_val_acc" : max(val_accs),     # best across epochs
        "num_params"   : num_params,
        "train_time"   : train_time,
        "train_losses" : train_losses,
        "val_accs"     : val_accs,
    }
