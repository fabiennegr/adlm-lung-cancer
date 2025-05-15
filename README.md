# Lung Cancer Risk Prediction Using Multimodal Deep Learning
###### <h6 align="right"> - A group project by Julian Link, Prasanga Dhungel, Fabienne Greier from TU - Munich</h6>

## Table of Contents

1. [Introduction](#introduction)
2. [Abstract](#abstract)
3. [Run Locally](#run-locally)
4. [Execution on a Server](#execution-on-a-server)
5. [Project Structure](#project-structure)

---

## Introduction

In the winter semester of 2023/2024, we took a course at TU Munich called [Practical Course: Applied Deep Learning in Medicine](https://aim-lab.io/theses/practical/). As part of this course, we were assigned a project that focused on developing a deep learning framework for lung cancer diagnosis using low-dose CT scans and associated clinical metadata. The objective was to explore and evaluate the effectiveness of multimodal learning approaches in enhancing malignancy prediction.

## Abstract

Lung cancer is the leading cause of cancer-related mortality worldwide. For its effective treatment, it is crucial to catch it at an early stage. Lung cancer screening, employing  low-dose Computed Tomography (CT) scans, has demonstrated considerable efficacy in decreasing mortality rates associated with lung cancer by identifying malignant pulmonary nodules at an earlier phase. In addition, Machine Learning and Deep learning approaches have shown promise in classifying lung cancer from Lung CT. However, most of the existing approaches are fundamentally limited as they fail to capture the information  provided by the metadata of the patients. Multimodal learning generally outperforms single-modality models in disease diagnosis and prediction. This is particularly true in lung cancer, which is heavily contextualized through non-imaging risk factors.

In this work, we show that combining different levels of features, including clinical metadata and imaging data at the lung and nodule levels, provides a good estimation of malignancy, surpassing the predictive capability of utilizing these data sources independently. Furthermore, our analysis reveals superior performance when employing a feature extractor pretrained specifically on lung CTs compared to one pretrained on alternative domains.

## Project Overview

### Pre-processing Pipeline
The outline of our pre-processing pipeline involves the identification of candidate nodules, followed by the selection of the top five most confident nodules situated within the lung region.

<p align="center">
  <img src="../main/documentation/Preprocessing.png" width="500" alt="Pre-processing Pipeline">
</p>

### Training Pipeline
For our Multimodal Approach, we integrate this nodule-level data with lung-level and clinical metadata. Feature extraction from nodules and lung images is accomplished using MedicalNet and Pretrained ResNet 3D, respectively, while neural networks are employed to embed the metadata.

<p align="center">
  <img src="../main/documentation/Training_Pipeline.png" width="500" alt="Training Pipeline">
</p>

## Run Locally

### Setting Up
Python version >= 3.10 is recommended.

```bash
pip install -r requirements.txt
```

#### Log in to Wandb
To log in to wandb, execute:

```bash
wandb login
```

When prompted, enter the API key that can be found [in your profile](https://wandb.ai/authorize).


Once all the necessary packages are installed and you are logged in with your wandb credentials, you can initiate the training of the network by executing:

```bash
python src/train_multimodal.py
```

Instead of train_multimodal.py, you can also run other files in the `src` directory. You can also change the hyperparameters in the `configs` directory.

## Execution on a Server
To dispatch your task to the job scheduler, execute:

```bash
sbatch scripts/submit_job.sh
```

This will provide you with a job-id. You can monitor the status of your job using this job-id with the following command:

```bash
scontrol show job **your-job-id**
```

The outputs of the tasks will be stored at `logs/slurm/your-job-id.out` and `logs/slurm/your-job-id.err`. You can also monitor the status of your job using the wandb dashboard.

### Preprocessing
The code for preprocessing resides in the `scripts` directory. For instance, dico2nifti_tcia.py is the code for converting the DICOM files to NIFTI format. The preprocessing code can be run independently of the main training code. To run the preprocessing code, simply execute:

```bash
python scripts/file-name.py
```

## Project Structure

```
.
├── configs
├── data
├── documentation
├── Makefile
├── notebooks
├── README.md
├── requirements.txt
├── scripts
└── src
    |── data
    |── models
    |── utils
    |── train_multimodal.py

```

- `configs`: Contains the configuration files for the model and the dataset.
- `data`: Contains the dataset. Most of the data is confidential and hence not included in the repository. They can be accessed from the server.
- `documentation`: Contains detailed project documentation, including the project report and final poster
- `scripts`: Contains the scripts used for the job submission to the server, and preprocessing codes.
- `src`: Contains the source code for the project. The `train_multimodal.py` file is the main file for training the model. Inside the `src` directory, the `data` directory contains the lightning data modules to load the data, the `models` directory contains the model and code for the forward pass, and the `utils` directory contains the utility functions used in our work.
- `notebooks`: Contains the notebooks used for the exploratory data analysis and some pre-processing steps.

##### Most of the data and some of the code are confidential and hence not included in the repository.
