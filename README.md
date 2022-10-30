# Udacity---Object-Detection-in-an-Urban-Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
	- training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```
The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.
```
You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

## ## MyGoal
This project is to use the SSD-ResNet-50 to train the model (consider as transfer learning) to detect and localize the vehicle, pedestrian and cyclist in urban environment. At the same time, it is needed to make the adjustment in the hyperparameters to make the model detect the objects well.

## Initial Workflow
At first, I try to setup in my own workspace by downloading the dataset by using the 'download_process.py' script. If using the script, the dataset will be downloaded, processed and saved in processed file. After that, by using the 'create_splits.py' script, I have made the code for splitting the dataset to ratio of 8:1:1 (training, testing and validating). However, due to the difficulty of environement setup, I cannot continue with EDA. Hence, I choose to use back the provided workspace. In the provided workspace, the data has been splitted well, therefore, it can be used directly with the EDA. In EDA, the project needs us to observe the characteristics of the dataset. 

## Dataset Observation
Here are some of results of annotation on the every objects based on the groundtruth bounding boxes provided in tfrecord file. The attached results show that the dataset consists of several environment such as sunnny/cloudy weather, day/night time. With the observation, it is noted that number of vehicles is the most all of the time in the dataset. 

![](https://file%2B.vscode-resource.vscode-cdn.net/Users/nazzainal/python_project/Udacity---Object-Detection-in-an-Urban-Environment/results/img1.png?version%3D1667094137470)