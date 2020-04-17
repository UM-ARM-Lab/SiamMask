# SiamMask ROS Version

## Run Message subscriber
```
rosrun SiamMask run_demo --resume SiamMask_DAVIS.pth --config config_davis.json

rostopic pub /SiamMask_BBox mps_msgs/AABBox2d "xmin: 50.0
ymin: 100.0
xmax: 100.0
ymax: 200.0"
```

## Run Action Server
```
rosrun SiamMask siamserver --resume SiamMask_DAVIS.pth --config config_davis.json

```

## Environment
```
export PYTHONPATH=/home/kunhuang/catkin_ws/devel/lib/python3/dist-packages:/home/kunhuang/.pyenv/versions/anaconda2-2019.03/envs/siammask/lib/python3.6/site-packages/cv2:/home/kunhuang/local/lib/python2.7/dist-packages:/home/kunhuang/mps_ws/devel/lib/python2.7/dist-packages:/home/kunhuang/armlab_ws/devel/lib/python2.7/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages

export ROS_PACKAGE_PATH=/home/kunhuang/mps_ws/src:/home/kunhuang/armlab_ws/src:/home/kunhuang/catkin_ws/src:/opt/ros/melodic/share
```

# DOCKER RELATED

## Usefull Docker Commands
```
docker run --gpus all -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY um_arm_lab/siam_mask:melodic

docker build . -f ci/Dockerfile -t um_arm_lab/siam_mask:melodic

docker run --rm -it um_arm_lab/siam_mask:melodic

docker start --attach -i lucid_jepsen

docker run --gpus all nvidia/cuda:10.0-base nvidia-smi

docker run --gpus all nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 nvidia-smi

docker pull nvidia/cuda:10.0
```

## Run demo.py
```
docker run --gpus all -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY um_arm_lab/siam_mask:melodic
```

```
source /root/catkin_ws/devel/setup.bash
export SIAM_MASK_PATH=/root/catkin_ws/src/siam_mask/src/SiamMask/

export PYTHONPATH=/root/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages:$SIAM_MASK_PATH/experiments/siammask_sharp:$SIAM_MASK_PATH

cd $SIAM_MASK_PATH/experiments/siammask_sharp

python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_davis.json
```
