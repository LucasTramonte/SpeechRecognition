# Speech Recognition Project
Developing a robust speech recognition system using state-of-the-art models and datasets.

---

<div style="display: flex; gap: 10px;">
  <a href="https://paperswithcode.com/dataset/librispeech">[Dataset]</a>
  <a href="https://paperswithcode.com/paper/conformer-based-target-speaker-automatic">[Conformer]</a>
</div>


## Experiments Under Consideration

Our project will be based on the paper: [Conformer: Convolution-augmented Transformer for Speech Recognition](https://paperswithcode.com/paper/conformer-based-target-speaker-automatic).


### Conformer: Convolution-augmented Transformer for Speech Recognition

This paper introduces the Conformer, a novel architecture that combines Convolutional Neural Networks (CNNs) and Transformers to achieve state-of-the-art performance in end-to-end Automatic Speech Recognition (ASR). The Conformer aims to model both local and global dependencies of audio sequences in a parameter-efficient manner, leveraging the strengths of CNNs for local feature extraction and Transformers for capturing long-range global interactions.

**Conformer Model Architecture:**
- **Conformer Block:** Combines:
  - Two Macaron-style Feed-Forward Modules sandwiching other components.
  - Multi-Head Self-Attention (MHSA) with relative positional embeddings for robust sequence modeling.
  - Convolution Module for efficient local feature extraction.
- The model processes input audio using a convolutional subsampling layer before feeding it into the Conformer blocks.


## Usage Instructions with [DCE](https://dce.pages.centralesupelec.fr/)

### GPU Access
To request a GPU session:
```bash
srun -p gpu_inter -t 00:30:00 --pty bash
```
To request a specific GPU node:

```bash
srun -p gpu_inter -t 00:30:00 --nodelist=sh03 --pty bash
```

Execute the main script as follows

```bash
python main.py
```

### NVIDIA NeMo Framework


```bash
pip install git+https://github.com/NVIDIA/NeMo-Run.git
```

```bash
pip install pip install hydra-core omegaconf
pip install nemo_toolkit['all']
```

## References

1. Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, and Ruoming Pang. "Conformer: Convolution-augmented transformer for speech recognition." In Interspeech 2020, pages 5036–5040, 2020.