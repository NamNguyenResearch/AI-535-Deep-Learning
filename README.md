# Universal Rate-Distortion-Classification Representations for Lossy Compression

<p align="center">
  <img src="https://imgur.com/Q1IVfHu.png" alt="Experimental Setup"/>
</p>

An illustration of the experimental setup for the universal model. A single encoder `f` is trained for an initial classification-distortion tradeoff and has its weights frozen. Subsequently, many other decoders `{g_i}` are optimized for different tradeoff points using the representations `z` produced by `f`. The sender and receiver have access to a shared source of randomness `u` for universal quantization. A pre-trained classifier (`C`) is provided.

### [[Paper]](https://www.overleaf.com/read/xpjpbqhpwmgh#6dfb9f)

## Overview

### Representation for Lossy Compression
The rate-distortion-classification tradeoff was observed as a result of applying pre-trained classifier regularization within deep-learning-based image compression [Wang2024]. In prior works, an entire end-to-end model is trained for each desired setting of rate, distortion, and classification. However, developing a new system from scratch for each objective is impractical. Instead, we aim to reuse trained networks with frozen weights whenever possible.

It is of interest to assess the distortion and classification penalties incurred by such model reuse, particularly in scenarios where the encoder is fixed in advance. 

- **End-to-End Models:** The encoder and decoder are trained jointly for an objective.
- **Universal Models:** The encoder is fixed, and only the decoder is trained. Encoders for universal models are borrowed from end-to-end models.

Within the same dataset, universal models and end-to-end models differ only in the trainability of the encoder.

## Requirements
See `requirements.txt`. The code should work on most builds.

## Setup and Training

The design follows closely the framework of Blau & Michaeli [Blau2019], George Zhang, et.al. [George2021] and Wang, et.al. [Wang2024]. We first train **end-to-end models**, where both `f` (encoder) and `g` (decoder) are trainable. 

- **Loss functions:**
  - Mean Squared Error (MSE) for distortion
  - Classification loss via a pre-trained classifier `c`

Reconstruction `\hat{x}` is fed to a pre-trained classifier (`c`) to obtain a probability vector `\hat{s}` representing class probabilities. The true label `s` and predicted `\hat{s}` are used to compute classification accuracy.

### Conditional Entropy Constraints

To track the posterior distribution `p_phi(s|\hat{x})`, we approximate it with a pre-trained classifier parameterized by `psi`:

```
H(S|\hat{X}) = \sum_{s}\sum_{\hat{x}} p_phi(s,\hat{x}) log(1 / p_phi(s|\hat{x}))
```

Using the KL-divergence property, we derive an upper bound for the conditional entropy constraint:

```
H(S|\hat{X}) <= CE(s, \hat{s})
```

where `CE(s, \hat{s})` is the cross-entropy loss.

### Rate Constraint
To control the compression rate `R`, we use the upper bound:
```
R = dim * log2(L)
```
where `dim` is the encoder output dimension, and `L` is the number of quantization levels per entry. As discussed in [Blau2019], setting `R` to its upper bound simplifies the scheme with minimal loss of performance [Agustsson2019].

### Loss Function
The overall loss function for lossy compression is:
```
L = E[||X - \hat{X}||^2] + lambda * CE(s, \hat{s})
```
where `g` acts as both a decoder and a generator, optimizing for both low distortion and low classification loss.

### Universal Models

After training the end-to-end models, their encoders are reused to construct **universal models**. The encoder `f` is frozen, and a new decoder `g_1` is trained with:

```
L_1 = E[||X - \hat{X}_1||^2] + lambda_1 * CE(s, \hat{s})
```

where `lambda_1` is another tradeoff parameter. The weights of `g_1` are randomly initialized, and training follows the same procedure as before. This process is repeated with different `lambda_i` values to generate a tradeoff curve.

## References

- Blau, Y., & Michaeli, T. (2019). *Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff*.
- George Zhang, et.al., (2021). *Universal Rate-Distortion-Perception Representations for Lossy Compression*.
- Wang, X. et.al., (2024). *Lossy Compression with Data, Perception, and  Classification Constraints*.
