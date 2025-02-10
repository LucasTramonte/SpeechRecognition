# Speech Recognition Project
Developing a robust speech recognition system using state-of-the-art models and datasets.

---

<div style="display: flex; gap: 10px;">
  <a href="https://paperswithcode.com/dataset/librispeech">[Dataset]</a>
  <a href="https://paperswithcode.com/paper/listen-attend-and-spell">[LAS]</a>
  <a href="https://paperswithcode.com/paper/conformer-based-target-speaker-automatic">[Conformer]</a>
</div>


## Experiments Under Consideration

Our project will be based on two key papers: [Conformer: Convolution-augmented Transformer for Speech Recognition](https://paperswithcode.com/paper/conformer-based-target-speaker-automatic) and [Listen, Attend and Spell](https://paperswithcode.com/paper/listen-attend-and-spell).


### Conformer: Convolution-augmented Transformer for Speech Recognition

This paper introduces the Conformer, a novel architecture that combines Convolutional Neural Networks (CNNs) and Transformers to achieve state-of-the-art performance in end-to-end Automatic Speech Recognition (ASR). The Conformer aims to model both local and global dependencies of audio sequences in a parameter-efficient manner, leveraging the strengths of CNNs for local feature extraction and Transformers for capturing long-range global interactions.

**Conformer Model Architecture:**
- **Conformer Block:** Combines:
  - Two Macaron-style Feed-Forward Modules sandwiching other components.
  - Multi-Head Self-Attention (MHSA) with relative positional embeddings for robust sequence modeling.
  - Convolution Module for efficient local feature extraction.
- The model processes input audio using a convolutional subsampling layer before feeding it into the Conformer blocks.

### Listen, Attend and Spell (LAS)

This paper presents the Listen, Attend and Spell (LAS) model, a neural network architecture for end-to-end speech recognition that transcribes speech utterances into character sequences. Unlike traditional DNN-HMM systems, LAS jointly learns all components of the speech recognition pipeline without relying on phonemes or pronunciation dictionaries.

**LAS Model Architecture:**
- Consists of two main components:
  - **Listener:** A pyramidal recurrent neural network (RNN) encoder that converts speech signals into high-level features.
  - **Speller:** An attention-based RNN decoder that generates character sequences from the features provided by the Listener.
- Uses content-based attention to align audio features with output characters dynamically.
- Overcomes limitations of Connectionist Temporal Classification (CTC) by modeling dependencies between characters.


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

## References

1. William Chan, Navdeep Jaitly, Quoc V. Le, and Oriol Vinyals. "Listen, attend and spell: A neural network for large vocabulary conversational speech recognition." In ICASSP, pages 4960–4964. IEEE, 2016.
2. Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, and Ruoming Pang. "Conformer: Convolution-augmented transformer for speech recognition." In Interspeech 2020, pages 5036–5040, 2020.