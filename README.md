# NeuroRG
This repository contains code for learning and evaluating the methodology introduced in the "NeuroRG(가제)" paper.

The paper addresses recent advancements in microscopy and deep learning algorithms for biomedical image analysis, acknowledging the challenges faced in their practical applications. The study aims to overcome limitations in deep learning-based cellular image analysis by combining advanced techniques with microscopy and experiments to automatically generate a large dataset labeled with expert precision. This innovative approach eliminates the need for manual curation while maintaining model performance, allowing efficient assessment of dynamic cellular morphological changes, the degree of neuroinflammation, and the effects of potential pharmaceutical interventions.

![image](https://github.com/tempBiotech/RG/assets/118416128/4b754896-ff9e-4cfb-90f2-8fae9fe7bfc5)

![image](https://github.com/tempBiotech/RG/assets/118416128/63d8783b-f992-4d20-8714-7472aa8a5d64)


For the details, please visit our paper, "NeuroRG(가제)".

## Setup
All the code was run under Python 3.10.6. If using conda, our recommended settings are as follows:
```
> conda create -n neurorg python=3.10.6
> conda activate neurorg
(neurorg) > pip install -r dependencies.txt
```

## Data Availability
Cropped image data is available in the form of numpy arrays on Zenodo (+ URL need to be added), and ownership of all data is explicitly stated to belong to DR.NOAHBIOTECH.
