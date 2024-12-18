# NeuroRG
This repository contains codes for evaluating the methodology with the model introduced in the "Unveiling CNS Cell Morphology with Deep Learning: A Gateway to Anti-Inflammatory Compound Screening" paper, focusing on the training and evaluation of the core ensemble analysis model.

## Introduction
Comprehending the structure and behavior of cells is of paramount importance, particularly in the investigation of brain-related disorders. However, the analysis of the morphology and characteristics of brain cells poses significant challenges. Despite the utilization of advanced techniques such as deep learning, obstacles persist, including the requirement for substantial labeled data, the intricacy of detecting subtle cellular changes, and variations in experimental conditions.

Our research is dedicated to overcoming these challenges, specifically within the context of neuroinflammation, an inflammatory process occurring in the brain. By leveraging proprietary data and employing a deep learning approach, we have effectively scrutinized the morphologies of both nerve and immune cells, both under pathological conditions and subsequent to pharmacological interventions. This innovative methodology enhances our understanding of neuroinflammation and facilitates the evaluation of potential treatments, thereby addressing a void in the exploration of brain disorders and the advancement of novel pharmaceuticals.

## Results
Firstly, we propose a novel method to classify wells by dividing a single well into 15 images and classifying the well based on the ratios derived from each image. Secondly, we incorporate a modified Leave-One-Out Cross-Validation approach and an ensemble methodology to develop a model that accounts for batch effects across plates, the sample groups in this study.
### Model & Accuracy
![git_fig1](https://github.com/user-attachments/assets/154cc434-89fb-4a04-acc3-fc8f23b564a3)

The above figures illustrate the accuracy benchmark metrics of the top-performing deep learning model and presents the image classification results for the 6 classes. This proposed framework demonstrates high performance in classifying wells.
### McNemar's Test for Model Comparision between Ensemble-based Model and Single Model
![git_fig2](https://github.com/user-attachments/assets/87fd48ad-6088-480b-bb5a-b26e9379c675)

Moreover, statistical analysis of performance differences with and without the ensemble approach revealed significant improvement with ensemble models. Notably, when analyzing the same samples, ensemble-based models correctly classified 32 samples exclusively, compared to just 1 sample for non-ensemble models. This substantial difference highlights the effectiveness of the ensemble approach in enhancing classification performance. 

These results underscore the utility of the proposed framework in addressing the inherent challenges of analyzing brain cell morphologies under varied experimental conditions.
For the details, please visit our paper, "Unveiling CNS Cell Morphology with Deep Learning: A Gateway to Anti-Inflammatory Compound Screening".

## Setup
Please download the image dataset saved in .npy format from Zenodo, reconfigure the data path in the src/driver.ipynb Jupyter notebook, and then execute the notebook.


All the code was run under Python 3.10.6. If using conda, our recommended settings are as follows:
```
> conda create -n neurorg python=3.10.6
> conda activate neurorg
(neurorg) > pip install -r dependencies.txt
```

## Data Availability
Cropped image data set is available in the form of numpy arrays on Zenodo (https://doi.org/10.5281/zenodo.10369052), and ownership of all data is explicitly stated to belong to DR.NOAHBIOTECH.
