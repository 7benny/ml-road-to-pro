# ML Road to Pro 🚀

A serious, hands-on journey to master **Deep Learning with PyTorch** — starting from first principles (tensors, neurons) all the way to implementing **CNNs, YOLO, and Transformers**.  
This repository tracks my progress level by level, with **theory + PyTorch code + practice exercises**.

---

## 🎯 Levels Roadmap

- [x] **Level 1 — Tensors & Neuron Math**  
  Basics of tensors (1D/2D/3D), dot products, and forward pass of a single neuron.

- [x] **Level 2 — Autograd & Gradient Descent**  
  Loss functions, `.backward()`, and how weights update through gradient descent.

- [ ] **Level 3 — Linear Regression (from scratch)**  
  Multiple training samples, full training loop, plotting loss curves.

- [ ] **Level 4 — Logistic Regression**  
  Binary classification, sigmoid, BCE loss, and evaluating accuracy/precision/recall.

- [ ] **Level 5 — Multi-Layer Perceptrons (MLPs)**  
  Hidden layers, ReLU activation, dropout, weight decay, and overfitting.

- [ ] **Level 6 — Training Craft**  
  Optimizers (SGD, Adam, Momentum), learning rate schedules, early stopping, checkpoints.

- [ ] **Level 7 — Convolutional Neural Networks (CNNs)**  
  Convolutions, strides, padding, pooling. Build a small CNN on images.

- [ ] **Level 8 — Full Training Pipeline**  
  Datasets, dataloaders, batching, metrics, reproducibility, experiment structure.

- [ ] **Beyond**  
  Transformers, NLP embeddings, YOLO object detection, re-implementing research papers.

---

## 📚 Structure

Each level includes:
1. **Concept** → Short explanation of theory.  
2. **Demo** → Minimal PyTorch code.  
3. **Practice** → Coding task or quiz to reinforce learning.  
4. **Takeaway** → Key lesson that scales up to real-world models.  

---

## 🛠️ Setup

Run everything in **Google Colab** (no local setup required).  
First cell to install dependencies:

```bash
!pip install torch torchvision torchaudio matplotlib tqdm
