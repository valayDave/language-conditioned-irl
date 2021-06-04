FROM tensorflow/tensorflow:2.3.0
RUN apt update && \
        DEBIAN_FRONTEND=noninteractive apt install -y \
        xvfb \
        libxrender1 \
        libcurl4-openssl-dev \
        libssl-dev \
        nano \
        python3-tk \
        wget \
        git \
        cmake \
        libxkbcommon-x11-0 locales gnupg2 lsb-release libeigen3-dev python3-sip-dev \ 
        netcat-openbsd && \
    mkdir -p /opt/home && \
    pip install \
        hashids \
        matplotlib \
        tensorflow-probability \
        tf-models-official \
        opencv-python \
        sklearn \
        pycurl \
        pyaml \
        lxml
ENV COPPELIASIM_ROOT=/opt/home/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/opt/home/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04
ENV QT_QPA_PLATFORM_PLUGIN_PATH=/opt/home/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04
# Get VRep
RUN cd /opt/home && wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04.tar.xz && \
    tar xvf CoppeliaSim_Edu_V4_0_0_Ubuntu18_04.tar.xz && \
    rm CoppeliaSim_Edu_V4_0_0_Ubuntu18_04.tar.xz
RUN cd /opt/home && git clone https://github.com/stepjam/PyRep.git && \
    cd PyRep  && git checkout 7057e19c6f2dfb72d2ab101705fb534bf98aa722 && pip install -r requirements.txt &&  python setup.py install
RUN cd /opt/home && git clone https://github.com/orocos/orocos_kinematics_dynamics.git && cd orocos_kinematics_dynamics && git reset --hard 1ae45bb && \
    cd /opt/home/orocos_kinematics_dynamics/orocos_kdl && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4 && make install && \
    cd /opt/home/orocos_kinematics_dynamics/python_orocos_kdl && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_VERSION=3 .. && make -j4 && cp PyKDL.so /usr/local/lib/python3.6/dist-packages

ADD ./language_conditioned_rl /opt/home/language_conditioned_irl/language_conditioned_rl
ADD ./GDrive /opt/home/language_conditioned_irl/GDrive
ADD setup.py /opt/home/language_conditioned_irl/setup.py
ADD requirements.txt /opt/home/language_conditioned_irl/requirements.txt
ADD Readme.md /opt/home/language_conditioned_irl/Readme.md
RUN cd /opt/home/language_conditioned_irl && pip install -r requirements.txt && python setup.py install
RUN pip install git+https://github.com/CMA-ES/pycma.git@master
ADD ./ros1compat /opt/home/language_conditioned_irl/ros1compat
ADD collect_data.py /opt/home/language_conditioned_irl/collect_data.py