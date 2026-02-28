# xai-oncology-analysis

🔬 Explainable AI in Oncology: Blood Cancer Cell Pattern Analysis

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)


![alt text](https://img.shields.io/badge/Python-3.8+-blue.svg)


![alt text](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

📝 Abstract

Early microscopic analysis of blood smears is essential in hematological oncology for understanding cancer cell morphology. This project presents an Explainable Artificial Intelligence (XAI)-driven framework for unsupervised deep learning–based analysis of blood cancer cell images.

Instead of relying solely on black-box labels, this system uses a Convolutional Autoencoder (CAE) to learn compact latent feature representations. These features are clustered via K-Means to identify distinct structural patterns. To ensure clinical transparency, the framework integrates:

Encoder Activation Heatmaps to highlight diagnostic regions.

Statistical Morphology Descriptors (Circularity, Solidity, etc.).

Dimensionality Reduction (PCA) for visual cluster verification.

🚀 Key Features

Unsupervised Representation Learning: Learns morphological features without human-labeled bias.

Explainable Pipeline: Moves beyond simple classification by providing heatmaps and geometric metrics.

Binary Categorization: Distinguishes between Cancer (Malignant Blast Cells) and Non-Cancer (Normal Hematopoietic Cells).

Research-First Design: Includes confidence estimation and silhouette scores for clustering quality.

📂 Dataset Structure

This project is designed to work with the C-NMC 2019 dataset. Ensure your data is organized as follows:

code
Text
download
content_copy
expand_less
/data
  /train
    /cancer       # (e.g., ALL / Blast cells)
    /non_cancer   # (e.g., HEM / Normal cells)
  /val
    /cancer
    /non_cancer
Recommended Sources:

C-NMC 2019: 15,000+ images for robust training.


🛠️ Tech Stack & Architecture

Core: Python 3.9+, PyTorch, Torchvision.

Feature Extraction: Convolutional Autoencoder (CAE) / EfficientNet-B0.

Clustering: Scikit-Learn (K-Means, PCA).

Morphometrics: Scikit-Image (RegionProps).

Visualization: Matplotlib, Seaborn.

💻 Installation & Setup

Clone the repository:

code
Bash
download
content_copy
expand_less
git clone https://github.com/your-username/xai-oncology-analysis.git
cd xai-oncology-analysis

Install dependencies:

code
Bash
download
content_copy
expand_less
pip install -r requirements.txt

Train the Model:

code
Bash
download
content_copy
expand_less
python train.py --epochs 20 --batch_size 32 --lr 1e-4
📊 Evaluation & XAI Metrics

The system evaluates performance using both traditional ML metrics and pathology-oriented descriptors:

Metric Type	Description
Classification	Precision, Recall, F1-Score, ROC-AUC.
Clustering	Silhouette Score, Centroid Distance.
Morphological	Area, Perimeter, Circularity, Solidity, Texture Variance.
XAI	Activation Heatmaps (Grad-CAM), PCA Feature Projections.
🔍 Inference & Prediction

To analyze a single blood smear image and generate an explainability report:

code
Python
download
content_copy
expand_less
from model import Predictor

# Initialize and predict
analyzer = Predictor(model_path='best_model.pth')
report = analyzer.run(img_path='sample_cell.jpg')

print(f"Prediction: {report['class']}")
print(f"Confidence: {report['confidence']}%")
print(f"Morphological Solidity: {report['solidity']}")
⚠️ Medical Disclaimer

STRICTLY FOR RESEARCH PURPOSES.
This software is an experimental framework for pattern analysis and is not a cleared medical device. It is not intended for clinical diagnosis, treatment, or medical decision-making. All findings should be verified by a qualified hematopathologist.

🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements in the feature extraction or explainability layers.

