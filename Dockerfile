# syntax=docker/dockerfile:1
FROM ros:noetic

# Making sure our ROS node has ports to connect through.
# These are the ports specified in `rospy.init_node()` in hardware.py
EXPOSE 45100
EXPOSE 45101

# 1. Install system dependencies.
# This layer rarely changes, so it caches perfectly.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update -y && apt-get install -y \
    python3 python3-pip git \
    ffmpeg libsm6 libxext6 ros-noetic-opencv-apps dos2unix

# 2. Copy ONLY the requirements file first.
COPY ./requirements.txt /requirements.txt
# 3. Install ALL Python dependencies in a single, cached step.
# If you need to add more packages later, just add them to the end of this command.
# Because this is above the code copy step, modifying your code won't trigger reinstalls!
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r /requirements.txt && rm /requirements.txt

COPY ./requirements-hardware.txt /requirements-hardware.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade "pip==24.3.1" \
    && python3 -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.4.1" \
    && python3 -m pip install -r /requirements-hardware.txt \
    && rm /requirements-hardware.txt

# 4. Set the working directory.
WORKDIR /root/catkin_ws

# 5. NOW copy the actual code.
# Any changes to your Python scripts will only invalidate the cache from this point downwards.
COPY ./catkin_ws .

# Set up the environment to actually run the code
COPY ./scripts/entrypoint.bash ./entrypoint.bash
COPY ./scripts/setup.bash ./setup.bash

# Convert the line endings for the Windows users,
# calling `dos2unix` on all files ending in `.py` or `.bash`
RUN find . -type f \( -name '*.py' -o -name '*.bash' \) -exec 'dos2unix' -l -- '{}' \; && apt-get --purge remove -y dos2unix && rm -rf /var/lib/apt/lists/*

# Compile the catkin_ws.
RUN bash -c 'source /opt/ros/noetic/setup.bash && catkin_make'

# Make scripts executable
RUN chmod -R u+x /root/catkin_ws/

# Uncomment these lines and comment out the last line for debugging
# RUN echo 'source /opt/ros/noetic/setup.bash' >> /root/.bashrc
# RUN echo 'source /root/catkin_ws/devel/setup.bash' >> /root/.bashrc
# RUN echo 'source /root/catkin_ws/setup.bash' >> /root/.bashrc

ENTRYPOINT ["./entrypoint.bash"]
