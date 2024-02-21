# NeuroRG
This repository contains codes for evaluating the methodology with the model introduced in the "NeuroRG(tentative title)" paper.

## Introduction
This paper emphasizes the importance of understanding cell behavior in relation to the intricate connections between cellular morphology and phenotypic changes in response to stimuli. It introduces recent studies highlighting the relevance of morphological features in cancer cell tumorigenicity and microglial activation states, proposing a novel methodology to explore the complex relationship between subtle cellular changes and phenotypic irregularities.

<img width="100%" alt="Fig5" src="https://github.com/tempBiotech/RG/assets/118416128/a5fbd296-52a7-4044-8e21-02c7aeb3ffd1">
The paper addresses recent advancements in microscopy and deep learning algorithms for biomedical image analysis, acknowledging the challenges faced in their practical applications. The study aims to overcome limitations in deep learning-based cellular image analysis by combining advanced techniques with microscopy and experiments to automatically generate a large dataset labeled with expert precision. This innovative approach eliminates the need for manual curation while maintaining model performance, allowing efficient assessment of dynamic cellular morphological changes, the degree of neuroinflammation, and the effects of potential pharmaceutical interventions.

<img width="100%" alt="Fig3" src="https://github.com/tempBiotech/RG/assets/118416128/abdf93ed-3f5b-4deb-b147-4ee192561c69">


For the details, please visit our paper, "NeuroRG(tentative title)".

## Setup
All the code was run under Python 3.10.6. If using conda, our recommended settings are as follows:
```
> conda create -n neurorg python=3.10.6
> conda activate neurorg
(neurorg) > pip install -r dependencies.txt
```

## Data Availability
Cropped image data and trained parameters are available in the form of numpy arrays on Zenodo (https://doi.org/10.5281/zenodo.10369052), and ownership of all data is explicitly stated to belong to DR.NOAHBIOTECH.
