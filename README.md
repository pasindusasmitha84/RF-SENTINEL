<div align="center">

# 🛡️ RF-SENTINEL
### Real-Time Deep Learning RF Fingerprinting & Physical-Layer Intrusion Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5%2FPyQtGraph-green.svg)](https://www.riverbankcomputing.com/software/pyqt5/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A high-performance physical-layer (PHY) security framework combining software-defined radio simulation, deep residual learning, and live analytical visualization.*

</div>

---

## 🌟 Overview

**RF-SENTINEL** is an advanced reference implementation designed to secure wireless communication channels at the physical layer. By leveraging deep learning fingerprinting and real-time SDR signal synthesis, it detects unauthorized transmitters, spoofing attacks, and high-power jamming events with low latency.

---

## 🚀 Key Architectural Pillars

* **⚡ Multi-Transmitter RF Generator:** Simulates concurrent QPSK transmission streams embedded with hardware-specific physical impairments (I/O gain/phase imbalance, DC offset, Saleh PA saturation, phase noise, and CFO), alongside dynamic spoofers and swept jammers.
* **🧠 1D Residual CNN Backend:** Employs an optimized PyTorch residual architecture (`Conv1d` + skip connections) processing raw $2 \times 256$ I/O frames to extract robust, L2-normalized 64-D device fingerprints.
* **🛡️ Security Decision Engine:** Implements real-time claimed-ID verification, cosine similarity thresholding against device prototypes, and Exponential Moving Average (EMA) smoothing for stable anomaly detection.
* **📊 Interactive Analytical GUI:** Powered by PyQt5 and PyQtGraph featuring real-time STFT waterfalls, Power Spectral Density (PSD) views, 2-D PCA latent space projections, and live security status consoles.

---

## 📦 Quick Start (Ubuntu)

### 1. Prerequisites & Dependencies
```bash
sudo apt update && sudo apt install -y libxcb-xinerama0 libxkbcommon-x11-0
pip install numpy torch pyqt5 pyqtgraph
