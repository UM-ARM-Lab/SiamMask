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
echo $PYTHONPATH
/home/kunhuang/catkin_ws/devel/lib/python3/dist-packages:/home/kunhuang/.pyenv/versions/anaconda2-2019.03/envs/siammask/lib/python3.6/site-packages/cv2:/home/kunhuang/local/lib/python2.7/dist-packages:/home/kunhuang/armlab_ws/devel/lib/python2.7/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages

echo $ROS_PACKAGE_PATH
/home/kunhuang/armlab_ws/src:/home/kunhuang/catkin_ws/src:/opt/ros/melodic/share
```