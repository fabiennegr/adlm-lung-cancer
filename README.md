# Lung Cancer Risk Prediction Using Multimodal Deep Learning

## Run Locally

### Setting Up
Python version >= 3.10 is recommended.

```bash
pip install -r requirements.txt
```

#### Login to Wandb
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
The code for preprocessing reside on the `scripts` directory. For instance dico2nifti_tcia.py is the code for converting the DICOM files to NIFTI format. The preprocessing code can be run independently of the main training code. To run the preprocessing code, simply execute:

```bash
python scripts/file-name.py
```

## Project Structure

```
.
├── configs
├── data
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
- `data`: Contains the dataset. Most of the data are confidential and hence not included in the repository. They can be accessed from the server.
- `scripts`: Contains the scripts used for the job submission to the server, and preprocessing codes.
- `src`: Contains the source code for the project. The `train_multimodal.py` file is the main file for training the model. Inside the `src` directory, the `data` directory contains the lightning data modules to load the data, the `models` directory contains the model, and code for forward pass, and the `utils` directory contains the utility functions used in our work.
- `notebooks`: Contains the notebooks used for the exploratory data analysis and some pre-processing steps.
