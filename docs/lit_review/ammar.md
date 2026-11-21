# Literature review — Ammar's part (DRAFT)

Covers: C1 ViT from scratch, C2 parameter-efficient fine-tuning.
Status: draft paragraph + source list. Verify each citation detail (venue, year, page numbers)
against the paper before it goes into Overleaf.

---

## Draft paragraph

**Vision Transformers in the small-data regime.** The Vision Transformer (ViT) of Dosovitskiy et al.
[1] splits an image into fixed-size patches, embeds each patch linearly, adds learned position
embeddings, and processes the resulting token sequence with a standard Transformer encoder. Unlike a
CNN, it has almost no built-in assumptions about images (no locality, no translation equivariance),
and the authors report that it only overtakes strong CNNs when pretrained on very large datasets such
as ImageNet-21k or JFT-300M. DeiT [2] showed that ViTs can be trained on ImageNet-1k alone by relying
on heavy data augmentation, regularization and distillation, and Steiner et al. [3] systematically
studied how augmentation and regularization trade off against dataset size for ViTs (their "AugReg"
recipe produced the checkpoint used in Challenge 2). Our Challenge 1 setting, with 100 training images,
is several orders of magnitude smaller than any of these, so we treat the course testbed's small ViT as
a baseline and ask how far small-data techniques can close the gap to CNNs and classical methods. Two
lines of work target exactly this. Lee, Lee and Song [4] propose Shifted Patch Tokenization (SPT),
which concatenates spatially shifted copies of the image before patching so each token sees a wider
neighbourhood, and Locality Self-Attention (LSA), which masks each token's attention to itself and
learns the softmax temperature so attention becomes sharper and more local; they report gains on
small datasets such as CIFAR and Tiny-ImageNet. Hassani et al. [5] replace the patch embedding with a
small convolutional tokenizer and the class token with attention-based sequence pooling (Compact
Convolutional Transformers), and show that such models can be trained from scratch on CIFAR-10
competitively with CNNs. Both add a locality bias back into the ViT, which is why we chose them as the
"beyond the course" extensions of our ViT baseline.

**Parameter-efficient fine-tuning.** With a pretrained backbone (Challenge 2), the question becomes how
many of its parameters to adapt. Training only a new linear head (the testbed's example) changes
nothing inside the backbone, while full fine-tuning updates every weight and risks overfitting 100
images. Parameter-efficient fine-tuning (PEFT) methods sit between these two extremes. BitFit [6]
trains only the bias terms of a pretrained Transformer (well under 1% of its parameters); on language
benchmarks it was competitive with full fine-tuning, particularly with small-to-medium training sets.
LoRA [7] freezes the pretrained weights and learns a low-rank update to selected weight matrices
(typically the attention projections), so the number of trainable parameters is controlled by the
rank. Visual Prompt Tuning (VPT) [8] keeps the whole backbone frozen and instead learns a small number
of extra input tokens, either only at the first layer ("shallow") or at every layer ("deep"); on the
few-shot VTAB-1k benchmark it often matched or beat full fine-tuning. We study these three because
they restrict adaptation in three different places (biases, weight matrices, inputs), which lets us
compare accuracy against the number of trained parameters and training time on the same backbone.

---

## Sources

[1] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at
    Scale," ICLR 2021. arXiv:2010.11929
[2] H. Touvron et al., "Training data-efficient image transformers & distillation through attention,"
    ICML 2021. arXiv:2012.12877
[3] A. Steiner et al., "How to train your ViT? Data, Augmentation, and Regularization in Vision
    Transformers," TMLR 2022. arXiv:2106.10270
[4] S. H. Lee, S. Lee, B. C. Song, "Vision Transformer for Small-Size Datasets," arXiv:2112.13492, 2021.
[5] A. Hassani et al., "Escaping the Big Data Paradigm with Compact Transformers," arXiv:2104.05704,
    2021.
[6] E. Ben Zaken, S. Ravfogel, Y. Goldberg, "BitFit: Simple Parameter-efficient Fine-tuning for
    Transformer-based Masked Language-models," ACL 2022. arXiv:2106.10199
[7] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.
    arXiv:2106.09685
[8] M. Jia et al., "Visual Prompt Tuning," ECCV 2022. arXiv:2203.12119

## Originality notes (for the originality statement)

- SPT + LSA [4] and the CCT tokenizer [5] (C1): small-data ViT techniques, not covered in the course.
- BitFit [6], LoRA [7], VPT [8] (C2): PEFT methods, not covered in the course.
- TODO: confirm against the course syllabus which of these were / were not covered.
