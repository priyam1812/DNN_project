# Visual Storytelling with Cross-Modal Learning

## Quick Links
- **[Experiments Notebook](CNN_Stroytelling_Model_Training.ipynb)** – Complete experimental workflow (training, evaluation, visualizations)
- **[Results – Figures](results/figures/)** – Training curves and qualitative visualizations
- **[Results – Tables](results/tables/metrics_summary.csv)** – Quantitative evaluation metrics

---

## Innovation Summary
**This project implements a cross-modal visual storytelling model that jointly learns from image sequences and textual context using CNN-based visual encoding and Transformer-based language modeling. The system aligns visual and textual representations to predict future visual embeddings and corresponding textual tokens, enabling coherent story progression across temporal frames.**

---

## Key Results

| Metric | Value |
|------|------|
| Final Training Loss | 0.067501077 |
| Text Top-1 Accuracy | 1.0 |
| Text Top-5 Accuracy | 1.0 |
| Image Cosine Similarity | 0.783047557 |

> All metrics are computed automatically after training and saved to the `results/tables/` directory.

---

## Most Important Finding
> The model successfully learns aligned visual–textual representations, as evidenced by decreasing training loss, meaningful image embedding similarity distributions, and confident top-k text predictions. This demonstrates the effectiveness of cross-modal learning for visual storytelling tasks.

Representative visualizations can be found in the `results/figures/` directory, including training loss curves and embedding similarity distributions.

---

## How to Reproduce

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Training Model:
   ```bash
   python src/train.py
   ```

3. After training completes, all models, figures, and evaluation tables will be saved automatically to the results/ directory.

Note: The project is fully compatible with Google Colab and local Python (≥3.10). Training time depends on GPU availability.

## Repository Structure:
project_root/
├── README.md
├── requirements.txt
├── CNN_Stroytelling_Model_Training.ipynb
├── config.yaml
├── src/
│   ├── model.py
│   ├── utils.py
│   ├── visualize.py
│   └── train.py
└── results/
    ├── figures/
    └── tables/

## Reproducibility & Good Practice

* All random seeds are fixed for reproducibility.
* Visualizations and metrics are generated programmatically.
* Model architecture, training, and evaluation are modularised for clarity and extensibility.

