# CarND-P03-BehaviorClone

CarND-P03-BehaviorClone implements a neural network model that clones
a driving behavior from user-run simulations.  The trained model
is used to autonomously drive the vehicle in the simulator.


## File Structure
### Project Requirements
- **[model.py](model.py)** - Python script that builds and trains the
    neural network model
- **[drive.py](drive.py)** - Python script that provides vehicle control
    to the simulator using the trained model
- **[model.h5](model.h5)** - Trained model that is used by `drive.py`
- **[writeup_report.md](writeup_report.md)** - Project write-up report
- **[video.mp4](video.mp4)** - Video recorded for a full lap driven in
    autonomous mode using the trained model

### Additional Files
- **[video.py](video.py)** - Convert recorded driving snapshots into mp4
    video
- **py-src**
    - **[graph_losses.py](py-src/graph_losses.py)** - Helper script that
        generates graphs for training and validation losses
- **results** - Project outputs such as plots and simulation images 

### Not Included
- **data** - Simulation images and driving logs.
[Sample Training Data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
can be downloaded and used.

## Getting Started
### [Download ZIP](https://github.com/gabeoh/CarND-P03-BehaviorClone/archive/master.zip) or Git Clone
```
git clone https://github.com/gabeoh/CarND-P03-BehaviorClone.git
```

### Setup Environment

You can set up the environment following
[CarND-Term1-Starter-Kit - Miniconda](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md).
This will install following packages required to run this application.

- Miniconda
- Python
- Jupyter Notebook

### Download Simulator
- [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
- [MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
- [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)


### Usage

#### Train Model
Train the neural network model using recorded driving simulation data.
```
$ python model.py [options]
```
##### Options
- **data_dir**: specify directory containing training data files
    - ex) `--data_dir='./data/'`
    - ex) `--data_dir='./data/trial01,./data/trial02'`
- **epochs**: number of epochs (default: 6)  
- **batch_size**: batch size (default: 128)
- **valid_split**: ratio of validation dataset
- **drop_rate**: drop rate on dropout layers

#### Run Drive Controller
Run drive controller to send control signals to simulator in autonomous
driving mode.
```
$ pythone drive.py model [image_folder]
```
##### Arguments
- **model**: trained model output file
- **image_folder** _(optional)_: specify output directory to store driving
    snapshots

#### Create Video
Convert recorded driving snapshots to mp4 video.
```
$ pythone video.py [--fps FPS] image_folder
```
##### Arguments
- **image_folder**: Path that contains driving snapshots
- **fps** _(optional)_: Frame per second (FPS) settings for the video
    (default: 60)
    - ex) `--fps 30` 


## License
Licensed under [MIT](LICENSE) License.
