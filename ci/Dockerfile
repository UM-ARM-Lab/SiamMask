FROM um_arm_lab/baseline:melodic

# TODO: Move this to baseline
# Install pip
RUN apt-get -yq update && \
    DEBIAN_FRONTEND=noninteractive apt-get -yqq install \
    python-pip && \
    rm -rf /var/lib/apt/lists/*

# Install python requirements of SiamMask
ADD src/SiamMask/requirements.txt /root
RUN pip install -r /root/requirements.txt

# Create a catkin workspace
RUN cd && \
    mkdir -p catkin_ws/src && cd catkin_ws/src && \
    source /opt/ros/$(rosversion -d)/setup.bash && \
    catkin_init_workspace

ADD ./ /root/catkin_ws/src/siam_mask