# Speech Recognition Project
Developing a robust speech recognition system using state-of-the-art models and datasets.

---

<div style="display: flex; gap: 10px;">
  <a href="https://paperswithcode.com/dataset/librispeech">[Dataset]</a>
  <a href="https://huggingface.co/deepl-project/conformer-finetunning">[Model]</a>
</div>

---

```plaintext
OpenFinanceAI/

├── Assets/             
├── Project/                
│   ├──audio/Datasets/     
│     ├── exemple/      
│     ├── Fine-tunning/    
│     ├── Test/     
│   ├──Models/      
│     ├── Conformer/  
│         ├──results/      
│            ├── fine-transcription_test.csv              #results from fine-tunning.py and inference.py
│            ├── transcription_differences.csv            #results from compare_transcription.py 
│            ├── transcription_test.csv                   #results from conformer_test.py 
│            ├── transcription_wer.csv                    #results from metrics.py 
│         ├── compare_transcription.py                    # compare the transcriptions between the original model and the fine-tunned one
│         ├── conformer_model.py                          # conformer model for all speakers
│         ├── conformer_test.py                           # conformer model for only one speaker
│         ├── fine-tunning.py            
│         ├── inference.py                                # use the fine-tunned model from hugging face
│         ├── metrics.py                                  # calculate WER for transcription_test.csv
│     ├── Whisper/  
│         ├──results/      
│            ├── transcription_test.csv                   #results from whisper_test.py
│            ├── transcription_wer.csv                    #results from metrics.py 
│         ├── metrics.py                                  # calculate WER for transcription_test.csv 
│         ├── whisper_model.py                            # whisper model for all speakers
│         ├── whisper_test.py                             # whisper model for only one speaker
├── .gitignore                             
├── README.md                                             # Project documentation
└── requirements.txt                                      # Python dependencies
```

## Experiments 

Our project will be based on the paper: [Conformer: Convolution-augmented Transformer for Speech Recognition](https://paperswithcode.com/paper/conformer-based-target-speaker-automatic) and 
[Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) .


### Conformer: Convolution-augmented Transformer for Speech Recognition

This paper introduces the Conformer, a novel architecture that combines Convolutional Neural Networks (CNNs) and Transformers to achieve state-of-the-art performance in end-to-end Automatic Speech Recognition (ASR). The Conformer aims to model both local and global dependencies of audio sequences in a parameter-efficient manner, leveraging the strengths of CNNs for local feature extraction and Transformers for capturing long-range global interactions.

**Conformer Model Architecture:**
- **Conformer Block:** Combines:
  - Two Macaron-style Feed-Forward Modules sandwiching other components.
  - Multi-Head Self-Attention (MHSA) with relative positional embeddings for robust sequence modeling.
  - Convolution Module for efficient local feature extraction.
- The model processes input audio using a convolutional subsampling layer before feeding it into the Conformer blocks.

### Robust Speech Recognition via Large-Scale Weak Supervision

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

2. Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and IlyaSutskever.   Robust speech recognition via large-scale weak supervision.arXiv preprintarXiv:2212.04356, 2022.