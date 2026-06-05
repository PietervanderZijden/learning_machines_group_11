# syntax=docker/dockerfile:1
FROM ros:noetic

# Makking sure our ROS node has ports to connect trough.
# These are the ports specified in `rospy.init_node()` in hardware.py
EXPOSE 45100
EXPOSE 45101

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update -y && apt-get install -y \
    python3 python3-pip git \
    ffmpeg libsm6 libxext6 ros-noetic-opencv-apps dos2unix

# Install dependencies.

# The python3 interpreter is already being shilled by ros:noetic, so no need for a venv.
COPY ./requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r /requirements.txt && rm /requirements.txt

# This cd's into a new `catkin_ws` directory anyone starting the shell will end up in.
WORKDIR /root/catkin_ws

# This copies the local catkin_ws into the docker container.
COPY ./catkin_ws .

# Set up the envoirement to actually run the code
COPY ./scripts/entrypoint.bash ./entrypoint.bash
COPY ./scripts/setup.bash ./setup.bash

# Convert the line endings for the Windows users,
# calling `dos2unix` on all files ending in `.py` or `.bash`
RUN find . -type f \( -name '*.py' -o -name '*.bash' \) -exec 'dos2unix' -l -- '{}' \; && apt-get --purge remove -y dos2unix && rm -rf /var/lib/apt/lists/*

# Compile the catkin_ws.
RUN bash -c 'source /opt/ros/noetic/setup.bash && catkin_make'

RUN chmod -R u+x /root/catkin_ws/

# Uncomment these lines and comment out the last line for debugging
# RUN echo 'source /opt/ros/noetic/setup.bash' >> /root/.bashrc
# RUN echo 'source /root/catkin_ws/devel/setup.bash' >> /root/.bashrc
# RUN echo 'source /root/catkin_ws/setup.bash' >> /root/.bashrc

ENTRYPOINT ["./entrypoint.bash"]
