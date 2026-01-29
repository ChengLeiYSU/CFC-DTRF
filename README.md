# CFC-DTRF
To address underwater imaging suffers from complex degradations, we propose the Cross-layer Feature Consistency-guided Dual-Transformer Reconstruction Framework (CFC-DTRF).

📌 CFC-DTRF: Cross-layer Feature Consistency-guided Dual-Transformer Reconstruction Framework for Underwater Image Enhancement

🔍 Background

Due to light scattering and absorption in underwater environments, underwater imaging often suffers from complex degradations such as color cast, blurring, and haze, which severely limit its effectiveness in applications such as ocean exploration, underwater robotics, and environmental monitoring.

🚀 Our Contributions

We propose CFC-DTRF, a dual-Transformer reconstruction framework guided by cross-layer feature consistency, which employs joint constraints in both the feature and pixel domains to effectively decouple content degradation from color degradation and significantly improve both reconstruction quality and computational efficiency. The main contributions are summarized as follows:
1. Feature-consistency supervision: a multi-stage training scheme that enforces consistency between cross-layer features and visual details.
2. SWCA-Transformer (Sliding-Window Content-Attention Transformer): focuses on local detail fidelity and enhances texture sharpness.
3. MSCA-Transformer (Multi-Scale Color-Attention Transformer): performs multi-scale color correction to mitigate underwater color distortions.
4. Our method outperforms existing state-of-the-art approaches on multiple datasets, particularly in terms of detail preservation and color accuracy.

📈 Applications

1. Marine environmental monitoring
2. Underwater robotic navigation
3. Underwater scientific imaging and video enhancement

📝 Paper Information

Title: Cross-layer feature consistency and dual-transformer residual framework for underwater image enhancement

Journal: Engineering Applications of Artificial Intelligence, Vol. 167, 113972, 2026

Authors: Xinbin Li, Lei Cheng, Song Han, Jing Yang, Hui Dang, Muge Li

DOI: https://doi.org/10.1016/j.engappai.2026.113972

Paper Link: ScienceDirect

```bibtex
@article{LI2026113972,
  title = {Cross-layer feature consistency and dual-transformer residual framework for underwater image enhancement},
  journal = {Engineering Applications of Artificial Intelligence},
  volume = {167},
  pages = {113972},
  year = {2026},
  issn = {0952-1976},
  doi = {https://doi.org/10.1016/j.engappai.2026.113972},
  url = {https://www.sciencedirect.com/science/article/pii/S0952197626002538},
  author = {Xinbin Li and Lei Cheng and Song Han and Jing Yang and Hui Dang and Muge Li},
  keywords = {Underwater image enhancement, Transformer architecture, Color correction, Multi-level supervision}
}

