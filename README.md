# NeuroRG
This repository contains codes for evaluating the methodology with the model introduced in the "NeuroRG(tentative title)" paper.

## Introduction
Understanding how cells look and behave is crucial, especially in the study of brain-related disorders. However, analyzing the shapes and characteristics of brain cells has been challenging. Even with advanced techniques like deep learning, there are obstacles such as needing a lot of labeled data, difficulty in spotting subtle changes in cells, and variations in experimental conditions. 

Our research focuses on overcoming these challenges, specifically in the context of neuroinflammation (inflammation in the brain). We used our own data and a deep learning approach to effectively study the shapes of nerve and immune cells, both in unhealthy conditions and after using drugs. This new method helps us better understand neuroinflammation and makes it easier to test potential treatments, filling a gap in the study of brain disorders and the development of new drugs.

## Results
### Model & Accuracy
<img width="100%" alt="Fig3" src="https://github.com/tempBiotech/RG/assets/118416128/abdf93ed-3f5b-4deb-b147-4ee192561c69">

The above figures illustrate the accuracy benchmark metrics of the top-performing deep learning model and presents the image classification results for the 6 classes of the most efficient model, EfficientNetB5, in the form of a confusion matrix.

### Channel Comparison
<img width="100%" alt="Fig5" src="https://github.com/tempBiotech/RG/assets/118416128/a5fbd296-52a7-4044-8e21-02c7aeb3ffd1">

Additionally, the above images convey information about a similarity analysis using Pearson correlation. It compares predictions generated by combining three channels with those obtained from each individual channel. Each data point on the scatter plot represents the predicted probability of control for each well.

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
