# Data Collection For the Robot Experiments. 

The required files can be downloaded from [here](https://drive.google.com/uc?id=1hxHmeBEWxhaiIFYW4BKpatz_AFnmqNxt). This will create the `GDrive` folder necessary creating the main configurations for collecting data and robot specification
## Clean Way To Collect Data
- Use the Damn Docker File. Build Docker container in the following way. 
```sh 
docker build -t valaygaurang/pyrep-lgr:0.1 .
```

- Running Docker Container For Collecting Data
```sh
docker run -it -v "$(pwd)/freshly_collected_data":/opt/home/language_conditioned_irl/freshly_collected_data -w /opt/home/language_conditioned_irl/freshly_collected_data valaygaurang/pyrep-lgr:0.1 bash
```
- Inside Docker Container Run
```sh
xvfb-run python collect_data.py --num-demos 2000 --num-procs 3
```

## Painfull Useless Info For setup From Scratch
- Requires (`orocos_kinamtics_dynamics`)[https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/INSTALL.md#without-catkin]
- Follow To Letter : https://github.com/orocos/orocos_kinematics_dynamics/issues/115#issuecomment-363828002
- Fix in Imported Lib : https://github.com/ros/kdl_parser/issues/44#issuecomment-689672444
- also Install `sudo apt install libeigen3-dev libcppunit-dev python-sip-dev python3-sip-dev python-psutil python3-psutil python-future python3-future`
- 