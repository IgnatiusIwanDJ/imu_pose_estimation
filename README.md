# IMU Pose Estimator
Recent advancement in deep learning, has enable us to formulate solution to complex problem such as classifying human activity. Inertial Measurement Unit (IMU) is sensor to measures orientation, velocity, and gravitational forces by combining Accelerometer, Gyroscope, and Magnetometer into one. By using MEMS technology, they are now more commonly seen as miniaturized sensors designed for easy integration. By using this sensors close to human body parts, we can predict human activity from sensor value changes. Previous methods include heavily engineered hand-crafted features extracted from noisy and abundant accelerometer data using signal-processing techniques and the such. With deep learning, we can input log data from sensor measurements and let computer formulate the solution

## Summary
CNN classifier to predict human activity based on IMU input. Model consisted of 3 convolutional 1d operation and 2 fully connected layers. For training, the model also follows federated learning methods where we simulate edge devices with pysft <i>VirtualWorker</i>. Trained data will be sent to each devices and get called by using their pointer. In the training loop, each iteration will send model to train locally and get back updated parameters when done.
The model achieved impressive result, but it needs to be noted that there are only 6 label and input channels is not big. For complex cases this model might fail. Another alternative is to use CNN-LSTM to predict short and long term activity.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- Already Preprocessed data sets for modeling.
    │
    ├── env                <- Anaconda environment file to install packages
    ├── log                <- Trained data folder
    |   ├── checkpoint     <- Checkpoint location each time best model created, separated by timestamp
    |   |
    |   ├── train_log      <- event log location to visualize with tensorboard
    │
    ├── models             <- Trained data folder
    |   ├── trained_model  <- Location of trained model, separated by folder timestamp.
    │
    ├── result             <- JSON result location after evaluation
    ├── test               <- Folder location to place test dataset for evaluation
    |
    ├── pose_estimation_walkthrough.ipynb     <- Jupyter notebook walkthrough, for easy run 
    │
    |
    ├── dataloader.py      <- Dataset class to construct Pytorch dataset iterator for easy call and manipulation
    ├── evaluation.py      <- Evaluation script to test trained model (python evaluate.py)
    ├── model.py           <- Model template constructor
    ├── trainer.py         <- Train loop script for learning model using federated learning series (python trainer.py)
    └── util.py            <- General function to be shared with other files
  
  
## Getting started
To start training, you can calling python files from command line or use jupyter notebooks for easy run.

### Setting up dataset
For using your own dataset you can modify the script to your use. This project use modified dataset from <i>mHealthDroid: a novel framework for agile development of mobile health applications</i> (Banos et al. 2014) and i don't own the right to distribute it. You can check the link [here](https://github.com/mHealthTechnologies/mHealthDroid).

### Installing environment (Anaconda)
- First, you have to install anaconda/miniconda to use the yml file from env folder,
i suggest miniconda for fast install (https://docs.conda.io/en/latest/miniconda.html)

- open anaconda prompt(miniconda3) and go to the env directory from anaconda prompt(miniconda3) there is a file called "env.yml" import it using : 
```sh
conda env create -f env.yml
```
- after that look if the environment in prompt change name to "pose_predictor", otherwise type:
```sh
conda activate pose_predictor
```

### Jupyter notebooks
- For easy run, use jupyter notebook, in the imu_pose_estimation folder type:
```sh
jupyter notebook
```
- Wait for the window to show up in your browser
- Run all cells

### Training model
- open anaconda prompt(miniconda3) and make sure active env is "pose_predictor" go to face_classifier folder, run: 
```sh
python trainer.py -lr 0.01 -e 30 -b 20
```
- the command takes several parameter, if you want to customize look at the trainer.py

<b>NOTE</b>: if using checkpoint, the argument epoch be overwritten by saved value in checkpoint

- wait until a text mentioned "training end...."

- while training, it will log metrics and loss, if you want a visualization, you can use tensorboard, open another anaconda command prompt, activate pose_predictor (conda activate pose_predictor), then run the following format:
```sh
tensorboard --logdir=your_tensorboard_log_folder
```
example:
```
tensorboard --logdir=log/train_log/2020-10-08/23-45-55
```

### Evaluate model
- open anaconda prompt(miniconda3) and make sure active env is "pose_predictor" go to imu_pose_estimation folder and run script with :
```sh
python evaluate.py -m your_model_location
```
example :
```sh
python evaluate.py -m model/pose_predictor.pt
```
- the command takes several parameter, if you want to customize look at the evaluate.py

- after finished, it will log several metrics of performance and dump the result in JSON file

## Dependencies

- Python 3 <br/>
- Pytorch 1.4.0
- Pysft
- Numpy
- Scikit-learn

## More about Pysft
Pysft is an awesome on-going python packages that enables user like me to easily integrate federate learning into python. PySyft decouples private data from model training, this package also support Differential Privacy, and Encrypted Computation (like Multi-Party Computation (MPC) and Homomorphic Encryption (HE)) within the main Deep Learning frameworks like PyTorch and TensorFlow.  

<b>If you want to know more, check this </b>[link](https://github.com/OpenMined/PySyft).

## References
* Banos, O., Garcia, R., Holgado-Terriza, J.A., Damas, M., Pomares, H., Rojas, I., Saez, A., Villalonga, C.:mHealthDroid: a novel framework for agile development of mobile health applications. In: Proceedings of the 6th International Work-conference on Ambient Assisted Living an Active Ageing (IWAAL 2014), Belfast, United Kingdom, December 2-5 (2014).
* Lima, Wesllen Sousa, Eduardo Souto, Khalil El-Khatib, Roozbeh Jalali, and Joao Gama. 2019. “Human Activity Recognition Using Inertial Sensors in a Smartphone: An Overview.” Sensors 19 (14): 3213.
* Ordóñez, Francisco Javier, and Daniel Roggen. 2016. “Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition.” Sensors 16 (1): 115.

