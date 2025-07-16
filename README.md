# HyVAE-GAN: A Hybrid VAE-GAN Architecture for Realistic and Structured Image Generation

## Abstract

HyVAE-GAN is a deep generative architecture that combines the strengths of Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) to generate high-quality, realistic, and diverse images while preserving structured latent space representations. VAEs are known for learning interpretable latent spaces but often produce blurry outputs. GANs, in contrast, generate sharper images but suffer from instability and lack of diversity. This hybrid approach effectively bridges the gap between the two.

## Model Architecture

The HyVAE-GAN architecture consists of three key components:

- **Encoder**: Maps input images into a structured probabilistic latent space using the reparameterization trick. It outputs a mean and variance vector to enable smooth and regularized latent sampling.

- **Decoder / Generator**: Acts both as a VAE decoder and GAN generator. It reconstructs images from the sampled latent vector using transposed convolutions, progressively upsampling features into sharp, high-fidelity images.

- **Discriminator**: A binary classifier trained adversarially to distinguish real images from generated ones. It guides the generator to produce visually convincing outputs by providing adversarial feedback.

<img width="534" height="271" alt="HyVAEGAN drawio (1)" src="https://github.com/user-attachments/assets/aa1f0d6b-1a98-416a-b41a-f489d40cf20c" />


The model is trained using a composite loss function combining:

- Reconstruction loss (from VAE)
- KL divergence (for latent regularization)
- Adversarial loss (from GAN)

This integrated learning process ensures both realistic image synthesis and robust latent space representation.

## Results & Evaluation

The model was evaluated on the MNIST dataset and benchmarked against standard VAE and standalone GAN architectures. Key results include:

- **Higher Visual Fidelity**: HyVAE-GAN produced sharper and more realistic images than VAEs, while maintaining diversity compared to GANs.

- **Improved Metrics**:

  - **Fréchet Inception Distance (FID)**: 69.13 _(lower is better)_
  - **Inception Score (IS)**: 3.11 ± 0.13 _(higher is better)_

- **Discriminator Classification Metrics**:

  - Accuracy: **99.33% (Train)**, **99.21% (Test)**
  - F1-Score: **0.9933 (Train)**, **0.9921 (Test)**

- **Faster Training Time**: Due to dual feedback from reconstruction and adversarial loss, HyVAE-GAN achieved faster convergence than both standalone models.

These outcomes demonstrate that HyVAE-GAN is not only more effective in image generation but also more efficient and stable during training, making it suitable for real-world generative applications.
