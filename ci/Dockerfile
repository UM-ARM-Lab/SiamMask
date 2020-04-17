FROM um_arm_lab/baseline:melodic

# Install python requirements of SiamMask
ADD src/SiamMask/requirements.txt /root
RUN pip install -r /root/requirements.txt

# Create a catkin workspace
RUN cd && \
    mkdir -p catkin_ws/src && cd catkin_ws/src && \
    source /opt/ros/$(rosversion -d)/setup.bash && \
    catkin_init_workspace

RUN mkdir -p /root/catkin_ws/src/siam_mask/src/SiamMask/experiments/siammask_sharp && \
    cd /root/catkin_ws/src/siam_mask/src/SiamMask/experiments/siammask_sharp && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
ADD ./ /root/catkin_ws/src/siam_mask
RUN cd /root/catkin_ws && \
    source /opt/ros/$(rosversion -d)/setup.bash && \
    catkin_make && \
    source devel/setup.bash && \
    cd /root/catkin_ws/src/siam_mask/src/SiamMask/ && \
    export SiamMask=$(pwd) && \
    bash make.sh && \
    export PYTHONPATH=/root/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages:$SiamMask/experiments/siammask_sharp:$SiamMask && \
    python2 -m pip install --upgrade torch future
