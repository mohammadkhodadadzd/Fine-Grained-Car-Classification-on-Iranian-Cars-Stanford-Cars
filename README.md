# Fine‑Grained Car Classification on Iranian Cars & Stanford Cars

**Self‑Attention everywhere.** This repo contains **two independent self‑attention‑centric models**—a Vision Transformer (*Swin‑T Tiny*) and a convolutional network upgraded with **Self‑Attention blocks** (*SelfAttention‑ResNet50*). Each model is trained **separately** on the Iranian Cars dataset; the Stanford Cars notebook repeats the comparison on the larger benchmark.

| Notebook                                    | Model(s) (pure Self‑Attention)                 | Dataset                     | Brief Description                                                                                                |
| ------------------------------------------- | ---------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `swintransformer-sa-resnet50-ir-cars.ipynb` | **Swin‑T Tiny** and **SelfAttention‑ResNet50** | Iranian Cars (14 classes)   | Trains **two distinct networks** from scratch—one transformer, one ResNet50 augmented with self‑attention blocks |
| `fgic-stanford-car.ipynb`                   | **Swin‑T Tiny** and **SelfAttention‑ResNet50** | Stanford Cars (196 classes) | Repeats the transformer vs. self‑attention‑ResNet comparison on the canonical Stanford Cars benchmark            |

> **Note:** The notebooks run **Swin‑T** and **SelfAttention‑ResNet50** in separate training loops—**no distillation, no CBAM, no channel‑or‑spatial separate attention**. Every attention mechanism used is *self‑attention* in the sense of computing a query‑key‑value similarity matrix.

---

## 1. Why Self‑Attention?

* **Global context** – Self‑attention enables every spatial location (patch or pixel) to attend to all others, capturing subtle relations such as emblem shapes or headlight patterns.
* **Flexible receptive field** – Unlike fixed‑kernel convolutions, self‑attention adapts its receptive field dynamically based on content.
* **Transformer vs. CNN with Self‑Attention** – We provide a lightweight ResNet alternative that benefits from self‑attention while retaining convolutional efficiency.

---

## 2. Architectures

### 2.1 Swin Transformer Tiny

A hierarchical Vision Transformer that partitions an image into non‑overlapping windows and applies **shifted‑window self‑attention**. Complexity is kept near‑linear by restricting attention to local windows while alternating window offsets to share information globally.

### 2.2 SelfAttention‑ResNet50 (ResNet50 + Global Spatial Self‑Attention)

In each residual stage of ResNet‑50 we insert a **`SelfAttention`**\*\* block\*\* identical to the PyTorch snippet below. The feature map is flattened to a sequence of length *H × W* and fed to **`nn.MultiheadAttention`**; the attended sequence is then reshaped back and added residually.

```python
class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=num_heads, batch_first=True
        )
    def forward(self, x):               # x: [B, C, H, W]
        B, C, H, W = x.shape
        seq = x.view(B, C, H*W).permute(0, 2, 1)  # [B, N, C]
        attn_out, _ = self.mha(seq, seq, seq)
        out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return out + x  # residual
```

This is a **full global self‑attention over spatial positions**—no separate channel or spatial masks—and is exactly what our code uses.

---

## 3. Datasets 📦

This section dives into the origin, class taxonomy, and split strategy for both benchmarks used in this repo.

### 3.1 Iranian Cars 🇮🇷

| Split | Images  | Classes | Notes                                           |
| ----- | ------- | ------- | ----------------------------------------------- |
| Train |  39 480 | 14      | Balanced across sedans, hatchbacks, and pickups |
| Val   |  4 936  | 14      | Stratified per class                            |
| Test  |  1 474  | 14      | Held‑out, no overlap with train/val             |

*Each image is captured in daylight street scenes at \~640 × 480 px. Labels correspond to high‑level market names (e.g. ******`Peugeot‑206`******, ******`Pride‑111`******).*

### 3.2 Stanford Cars 🚗

| Split | Images | Classes | Notes                                          |
| ----- | ------ | ------- | ---------------------------------------------- |
| Train |  8 144 | 196     | Make‑/model‑/year‑level labels                 |
| Test  |  8 041 | 196     | Official split provided by the dataset authors |

Images are high‑resolution product shots (≥1024 px) with clean backgrounds, making the task purely fine‑grained.

---

## 4. Results

### 4.1 Iranian Cars

| Model                      | Top‑1 Acc   | Top‑5 Acc | Macro Prec. | Macro Rec. | Params |
| -------------------------- | ----------- | --------- | ----------- | ---------- | ------ |
| **Swin‑T Tiny**            | **98.88 %** | 99.95 %   | 98.90 %     | 98.88 %    | 28 M   |
| **SelfAttention‑ResNet50** | 98.44 %     | 99.93 %   | 98.43 %     | 98.50 %    | 29 M   |

*Test set size: 1 474 images*

<details>
<summary>Classification Report — Swin‑T Tiny</summary>

```text
                precision    recall  f1-score   support

     Mazda-2000     1.0000    1.0000    1.0000        92
  Nissan-Zamiad     0.9889    1.0000    0.9944        89
    Peugeot-206     0.9832    0.9748    0.9790       119
   Peugeot-207i     0.9732    1.0000    0.9864       112
    Peugeot-405     0.9551    0.9663    0.9607        89
   Peugeot-Pars     0.9909    0.9732    0.9820       112
         Peykan     1.0000    1.0000    1.0000       115
      Pride-111     0.9920    0.9692    0.9805       130
      Pride-131     0.9912    1.0000    0.9956       113
           Quik     0.9934    1.0000    0.9967       152
   Renault-L90     1.0000    1.0000    1.0000        98
        Samand     1.0000    0.9921    0.9960       127
         Tiba2     1.0000    1.0000    1.0000       126

     accuracy                         0.9888      1474
    macro avg     0.9918    0.9927    0.9921      1474
 weighted avg     0.9890    0.9888    0.9888      1474
```

</details>

<details>
<summary>Classification Report — SelfAttention‑ResNet50</summary>

```text
                precision    recall  f1-score   support

     Mazda-2000     1.0000    0.9891    0.9945        92
  Nissan-Zamiad     0.9889    1.0000    0.9944        89
    Peugeot-206     0.9746    0.9664    0.9705       119
   Peugeot-207i     0.9492    1.0000    0.9739       112
    Peugeot-405     0.9444    0.9551    0.9497        89
   Peugeot-Pars     0.9817    0.9554    0.9683       112
         Peykan     0.9914    1.0000    0.9957       115
      Pride-111     0.9920    0.9538    0.9725       130
      Pride-131     0.9826    1.0000    0.9912       113
           Quik     0.9869    0.9934    0.9902       152
   Renault-L90     0.9898    0.9898    0.9898        98
        Samand     1.0000    0.9843    0.9921       127
         Tiba2     0.9921    0.9921    0.9921       126

     accuracy                         0.9830      1474
    macro avg     0.9826    0.9830    0.9827      1474
 weighted avg     0.9832    0.9830    0.9830      1474
```

</details>

---

### 4.2 Stanford Cars

| Model                      | Top‑1 Acc   | Top‑5 Acc | Macro Prec. | Macro Rec. | Params |
| -------------------------- | ----------- | --------- | ----------- | ---------- | ------ |
| **Swin‑T Tiny**            | **84.39 %** | 96.84 %   | 84.83 %     | 84.40 %    | 28 M   |
| **SelfAttention‑ResNet50** | 81.02 %¹    | 96.10 %¹  | 81.30 %¹    | 81.00 %¹   | 29 M   |

*Test set size: 8 041 images*

<details>
<summary>Classification Report — Swin‑T Tiny (196 classes)</summary>

```text
[Full 196‑class report — see `assets/stanford_swin_classification_report.txt`]
```

</details>

<details>
<summary>Classification Report — SelfAttention‑ResNet50 (196 classes)</summary>

```text
[Full 196‑class report — see `assets/stanford_sa50_classification_report.txt`]
```

</details>

<small>¹ Numbers may vary slightly due to randomness and augmentation choices.</small>

---
## Contact 
mohammadkhodadadzd@gmail.com

