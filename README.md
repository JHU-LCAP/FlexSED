# FlexSED: Towards Open-Vocabulary Sound Event Detection

[![arXiv](https://img.shields.io/badge/arXiv-2409.10819-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2509.18606)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/Higobeatz/FlexSED/tree/main)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/OpenSound/FlexSED)

FlexSED is an easy-to-use, open-vocabulary sound event detection (SED) system. It can be used for data annotation, labeling, and developing evaluation metrics for audio generation.

## News
- Oct 2025: 📦 Released code and pretrained checkpoint  
- Sep 2025: 🎉 FlexSED Spotlighted at WASPAA 2025


## Installation

Clone the repository:
```
git clone https://github.com/JHU-LCAP/FlexSED.git 
```
Install the dependencies:
```
cd FlexSED
pip install -r requirements.txt
```

## Usage
```python
from api import FlexSED
import torch
import soundfile as sf

# load model
flexsed = FlexSED(device='cuda')

# run inference
events = ["Door", "Male Speech", "Laughter", "Dog"]
preds = flexsed.run_inference("example.wav", events)

# visualize prediciton
flexsed.to_multi_plot(preds, events, fname="example")

# (Optional) visualize prediciton by video
# flexsed.to_multi_video(preds, events, audio_path="example.wav", fname="example")
```

## Training

1. **Download** the AudioSet-Strong subset. The archive is available under [WavCaps](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/Zip_files/AudioSet_SL) and [HF-AS-Strong](https://huggingface.co/datasets/enyoukai/AudioSet-Strong)

2. **Prepare metadata** following the preprocessing steps. Feel free to check processed [metadata](https://huggingface.co/Higobeatz/FlexSED/tree/main/meta_data).

   (If you wish to create a validation split, remove a subset of samples from the training metadata and format them the same as the test metadata. Recommended: ~2000 samples across ~50 sound classes.)

4. **Update file paths** for both metadata and audio in `src/configs`.

5. **Extract CLAP embeddings**
   ```bash
   python src/prepare_clap.py
   ```
6. **Run training:**
   ```bash
   python src/train.py
   ```

## Reference

If you find the code useful for your research, please consider citing:

```bibtex
@article{hai2025flexsed,
  title={FlexSED: Towards Open-Vocabulary Sound Event Detection},
  author={Hai, Jiarui and Wang, Helin and Guo, Weizhe and Elhilali, Mounya},
  journal={arXiv preprint arXiv:2509.18606},
  year={2025}
}
```
