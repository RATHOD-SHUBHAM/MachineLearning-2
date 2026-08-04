# nn.Module and Training Step

## Say this (60–90 sec)
nn.Module is the base class for all PyTorch models. Subclass it, define layers in __init__, implement forward with the computation. Register layers as attributes — nn.Linear, nn.Conv2d — so parameters are discovered automatically. forward defines the pass; never call backward inside forward. A standard training step: set model.train(), load batch to device, zero gradients, forward pass, compute loss, backward, optimizer step. Optionally clip gradients, log metrics, step scheduler. Validation uses the same forward but no backward. Keep the loop clean — one function train_one_epoch, one validate. DataLoader handles batching and shuffle. Loss and optimizer are constructed outside the module. Parameters live in model.parameters() — pass to optimizer once. That's the entire PyTorch training skeleton in practice.

## Why it matters
The canonical training loop question. Shows you can implement end-to-end training without Hugging Face hiding the details.

## How it works
```python
model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
```
- **Module**: layers + forward; `model.parameters()` for optimizer.
- **DataLoader**: batch, shuffle, num_workers for parallel loading.
- **Loss**: task-specific — CE for classification, MSE for regression.

## Tradeoffs
- Use when: any custom model or fine-tuning loop where you control training.
- Avoid when: reinventing HF Trainer for standard LM fine-tune — use tools unless you need custom logic.

## If they dig deeper
- register_buffer vs Parameter — buffers saved in state_dict but not trained (e.g., running mean in BN).
- forward hooks — debug activations without changing forward code.
- torch.nn.parallel.DistributedDataParallel — multi-GPU training wrapper.
