# Autoencoders

## Say this (60–90 sec)
An autoencoder learns to compress input into a low-dimensional latent code and reconstruct the original. Encoder maps input to bottleneck; decoder maps bottleneck back to reconstruction. Train by minimizing reconstruction loss — MSE for continuous data, BCE for binary inputs. The bottleneck forces the network to learn useful structure, not copy blindly. Use cases: dimensionality reduction like nonlinear PCA, denoising — corrupt input, reconstruct clean — anomaly detection — high reconstruction error means unusual sample. Variational autoencoders add a probabilistic latent space with KL regularization — generate new samples by sampling latent z. Undercomplete autoencoder: latent dim < input dim. Overcomplete with sparsity penalties also works. See also hands-on notebooks in [`Algorithm_from_Scratch/AutoEncoder/`](../../Algorithm_from_Scratch/AutoEncoder/) — basic AE and VAE examples from scratch.

## Why it matters
Foundation for representation learning, generative models, and anomaly detection. Links unsupervised learning to modern latent diffusion and VAE concepts.

## How it works
- **Architecture**: encoder `z = f(x)`, decoder `x̂ = g(z)`, minimize `||x - x̂||²`.
- **Bottleneck**: latent dim d << input dim — compression.
- **Denoising AE**: train on `x̃ = x + noise`, reconstruct x — robust features.
- **VAE**: encoder outputs μ, σ; sample z; KL loss pulls latent toward prior (often N(0,1)).
- **Anomaly detection**: train on normal data only; outliers have high reconstruction error.

## Tradeoffs
- Use when: unsupervised features, denoising, compression, anomaly detection on structured data.
- Avoid when: need sharp realistic generation — plain AE latent isn't smooth; use VAE, GAN, or diffusion.

## If they dig deeper
- Sparse autoencoders — L1 penalty on activations for interpretable features.
- Contractive autoencoder — penalize sensitivity to input perturbations.
- Tie weights — decoder weights as transpose of encoder — parameter efficiency.
