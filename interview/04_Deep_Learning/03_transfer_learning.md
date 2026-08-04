# Transfer Learning

## Say this (60–90 sec)
Transfer learning reuses a model trained on a large source task — usually ImageNet for vision or a big text corpus for language — and adapts it to your smaller target task. Instead of training from scratch, you start with learned features that already capture useful patterns. Two common patterns: feature extraction — freeze the backbone, train only a new head; and fine-tuning — unfreeze some or all layers and train with a smaller learning rate. When target data is small, freezing early layers prevents overfitting. When data is larger or domain is different, fine-tune deeper layers. Always use a lower LR for pretrained weights than the new head — often 10x lower. Data augmentation matters more with small sets. Transfer learning is why a team with 500 images can beat training a huge model from random init.

## Why it matters
Standard practice in industry — almost nobody trains ResNet from scratch on custom vision. Shows you know efficient deployment, not just architecture names.

## How it works
- **Pretrain**: large dataset, general task (classification, MLM, contrastive).
- **Replace head**: swap final layer for your num_classes or task-specific output.
- **Freeze**: `param.requires_grad = False` for early layers; train head only.
- **Fine-tune**: unfreeze top blocks or entire network; discriminative LR (lower for early layers).
- **Domain gap**: medical vs natural images may need more fine-tuning and augmentation.

## Tradeoffs
- Use when: limited labeled data, target task related to pretrain task, need fast iteration.
- Avoid when: target domain totally unrelated and tiny data — pretrain features may hurt; consider from-scratch or self-supervised on domain data.

## If they dig deeper
- Linear probing vs fine-tuning — probe tests representation quality without changing backbone.
- Catastrophic forgetting — aggressive fine-tune can erase pretrain knowledge; use small LR, regularization.
- NLP: BERT embeddings frozen vs full fine-tune; LLMs use LoRA instead of full weight updates.
