# PyTorch & ML/AI Interview Study Roadmap

Chapter 1 is complete: tensors, rank/shape, operations, `view`/`reshape`, and NumPy interop.

---

## Part A — PyTorch (Implementation Track)

| Ch | Topic | What to Learn | File Idea | Interview Focus |
|----|--------|---------------|-----------|-----------------|
| **1** ✅ | **Tensors** | Create, dtype, device, shape/rank, indexing, math, `view`/`reshape`, NumPy sharing | `chap_1_tensors.py` | Tensor vs ndarray, contiguous memory |
| **2** | **Autograd** | `requires_grad`, forward, `.backward()`, `grad`, computation graph, `detach`, `no_grad` | `chap_2_autograd.py` | What is a gradient? Why chain rule? `zero_grad()` |
| **3** | **Training loop** | `nn.Module`, loss, optimizer, epochs, batch; manual loop before Trainer | `chap_3_training_loop.py` | Full train step in 5 lines |
| **4** | **Datasets & DataLoaders** | `Dataset`, `DataLoader`, transforms, batching, shuffling, `num_workers` | `chap_4_dataloader.py` | Batch size trade-offs, I/O bottlenecks |
| **5** | **Device & performance** | `.to(device)`, GPU vs CPU, `pin_memory`, mixed precision basics | `chap_5_device.py` | When data copies happen |
| **6** | **nn modules** | `Linear`, activations, `Sequential`, parameters vs buffers, `state_dict` | `chap_6_nn_modules.py` | Parameters vs hyperparameters |
| **7** | **CNNs** | Conv2d, pooling, channels, image shape `(N,C,H,W)` | `chap_7_cnn.py` | Output size formula, receptive field |
| **8** | **Transfer learning** | Pretrained models, freeze layers, fine-tune head | `chap_8_transfer_learning.py` | When to freeze vs fine-tune |
| **9** | **RNNs / sequences** | LSTM/GRU, padding, pack_padded_sequence (basics) | `chap_9_rnn.py` | Vanishing gradients, seq length |
| **10** | **Saving & deployment** | `torch.save`, `load_state_dict`, inference mode | `chap_10_save_load.py` | Train vs eval mode |
| **11** | **Transformers (PyTorch)** | `nn.Transformer`, attention intuition, tokenization with HF (optional) | `chap_11_transformers.py` | Self-attention Q,K,V (high level) |
| **12** | **Debugging PyTorch** | Shape errors, `grad is None`, NaNs, `torch.autograd.set_detect_anomaly` | `chap_12_debugging.py` | Common failure modes |

**Convention:** One `.py` (or notebook) per chapter, runnable end-to-end. Reuse patterns from `Alorithm_from_Scratch` where helpful (e.g. linear regression → Ch 3).

---

## Part B — ML/DL Theory (Parallel Interview Track)

Study one theory block per week alongside PyTorch chapters.

| Block | Topics | Tie to PyTorch Ch |
|-------|--------|-------------------|
| **B1** | Linear regression, loss (MSE), gradient descent | Ch 2–3 |
| **B2** | Logistic regression, classification metrics (acc, precision, recall, F1, ROC-AUC) | Ch 3–4 |
| **B3** | Regularization (L1/L2), bias–variance, train/val/test split | Ch 3 |
| **B4** | Backprop, chain rule, vanishing/exploding gradients | Ch 2 |
| **B5** | Optimizers (SGD, momentum, Adam), LR schedules | Ch 3 |
| **B6** | Batch norm, dropout, weight init | Ch 6–7 |
| **B7** | CNN theory (kernels, pooling, architectures) | Ch 7–8 |
| **B8** | RNN/LSTM, attention, Transformers (high level) | Ch 9, 11 |
| **B9** | GenAI: tokens, embeddings, pretrain vs fine-tune, RLHF (overview) | Ch 11 |

Existing repo work to revisit after Ch 2–3: linear regression, autoencoder, and VAE notebooks under `Alorithm_from_Scratch/` — reimplement one model in pure PyTorch.

---

## Part C — GenAI / Applied AI (After Ch 6–7)

| Ch | Topic |
|----|--------|
| **G1** | Tokenization, embeddings, context length |
| **G2** | Hugging Face: `AutoModel`, `AutoTokenizer`, inference |
| **G3** | Fine-tuning (LoRA overview), PEFT |
| **G4** | RAG pipeline (retriever + LLM), eval |
| **G5** | Prompting, agents, tool use (system design) |

---

## Suggested Pace (8–10 Weeks)

| Week | PyTorch | Theory / Project |
|------|---------|------------------|
| 1 | Ch 1 ✅ — polish indexing/slicing | B1 |
| 2 | Ch 2 Autograd | B4 |
| 3 | Ch 3 Training loop | B2, B3 |
| 4 | Ch 4–5 DataLoader + device | B5 |
| 5 | Ch 6 nn modules | Rebuild linear reg in PyTorch |
| 6 | Ch 7 CNN | B6, B7 |
| 7 | Ch 8 Transfer learning | Small image classifier project |
| 8 | Ch 9–10 RNN + save/load | B8 |
| 9–10 | Ch 11–12 + G1–G2 | B9, mock interviews |
