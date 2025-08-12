#  Model Overview

## Description: 
A2SB uses a UNet architecture to perform inpainting on an audio spectrogram. It can fill in missing frequency bands above 4kHz (bandwidth extension), or fill in short temporal slices (currently supporting filling in gaps of less than 1 second). This model is for non commercial use only.

### License/Terms of Use:
The model is provided under the NVIDIA OneWay NonCommercial License. 

The code is under [NVIDIA Source Code License - Non Commercial](https://github.com/NVlabs/I2SB/blob/master/LICENSE). Some components are adapted from other sources. The training code is adapted from [I2SB](https://github.com/NVlabs/I2SB) under the [NVIDIA Source Code License - Non Commercial](https://github.com/NVlabs/I2SB/blob/master/LICENSE). The model architecture is adapted from [Improved Diffusion](https://github.com/openai/improved-diffusion/blob/main/LICENSE) under the MIT License. 

### Deployment Geography:
Global

### Use Case:
Research purposes pertaining to audio enhancement and generative modeling, as well as for general creative use such as bandwidth extension and inpainting short segments of missing audio.

### Release Date:
Github 06/27/2025 via github.com/NVIDIA/diffusion-audio-restoration

## Reference(s):
- [project page](https://research.nvidia.com/labs/adlr/A2SB)
- [technical report](https://arxiv.org/abs/2501.11311)
- [I2SB](https://github.com/NVlabs/I2SB)
- [Improved-Diffusion UNet Architecture](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py) 


## Model Architecture:
**Architecture Type:** CNN with interleaved Self-Attention Layers

**Network Architecture:** UNET



## Input: 
**Input Type(s):** Audio

**Input Format(s):** WAV/MP3/FLAC

**Input Parameters:** One-Dimensional (1D)

**Other Properties Related to Input:** All audio assumed to be single-channeled, 44.1kHz. For editing, also provide frequency cutoff for bandwidth extension sampling (resample content above this frequency), or start/end time stamps for segment inpainting.

## Output: 
**Output Type(s):** Audio

**Output Format(s):** WAV

**Output Parameters:** One-Dimensional (1D)

**Other Properties Related to Output:** Single-channeled 44.1kHz output file. Maximum audio output length is 1 hour.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):**
* [PyTorch-2.2.2+cuda12.1+cudnn8]


**Supported Hardware Microarchitecture Compatibility:**
* NVIDIA Ampere
* NVIDIA Blackwell 
* NVIDIA Jetson  
* NVIDIA Hopper 
* NVIDIA Lovelace 
* NVIDIA Pascal
* NVIDIA Turing 
* NVIDIA Volta


**[Preferred/Supported] Operating System(s):**
['Linux']

## Model Versions:
v1

# Training and Evaluation Datasets:

## Training Datasets:

The property column below shows the total duration before license, quality, and sampling rate filtering. Our model training code ingests only raw audio samples -- no additional labels provided in the datasets listed below are used for training purposes.

| DatasetName | Collection Method | Labeling Method | Properties |
| ------ |  ------ | ------ | ------ | 
| [FMA](https://github.com/mdeff/fma) | Human | N/A | 5257.0 hrs |
| [Medleys-solos-DB](https://medleydb.weebly.com/) | Human | N/A | 17.8 hrs| 
| [MUSAN](https://www.openslr.org/17/)  | Human | N/A | 42.6 hrs | 
| [Musical Instrument](https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset) | Human| N/A | 16.2 hrs | 
| [MusicNet](https://zenodo.org/records/5120004) | Human | N/A | 34.5 hrs | 
| [Slakh](https://github.com/ethman/slakh-utils)  | Hybrid | N/A | 118.3 hrs| 
| [FreeSound](https://freesound.org/)  | Human | N/A | 4576.6 hrs| 
| [FSD50K](https://zenodo.org/records/4060432)  | Human | N/A | 75.6 hrs| 
| [GTZAN](http://marsyas.info/index.html)  | Human | N/A | 8.3 hrs| 
| [NSynth](https://magenta.tensorflow.org/datasets/nsynth)  | Human | N/A | 340.0 hrs| 


## Evaluation Datasets:
| DatasetName | Collection Method | Labeling Method | Properties |
| ------  | ------ | ------ | ------ | 
| [AAM: Artificial Audio Multitracks Dataset](https://zenodo.org/records/5794629) | Automated | N/A | 4 hrs | 
| [Maestro](https://magenta.tensorflow.org/datasets/maestro) | Human | N/A | 199.2 hrs | 
| [MTD](https://www.audiolabs-erlangen.de/resources/MIR/MTD) | Human | N/A | 0.9 hrs | 
| [CC-Mixter](https://members.loria.fr/ALiutkus/kam/) | Human | N/A | 3.2 hrs | 
 

## Inference:
**Engine:** PyTorch

**Test Hardware:**
* NVIDIA Ampere

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).