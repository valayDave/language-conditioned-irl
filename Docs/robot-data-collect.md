# Data Collection For the Robot Experiments. 

## Clean Way To Collect Data
- Use the Damn Docker File. Build Docker container in the following way. 
```sh 
docker build -t valaygaurang/pyrep-lgr:0.1 .
```

- Running Docker Container For Collecting Data
```sh
docker run -it -v "$(pwd)/freshly_collected_data":/opt/home/language_conditioned_irl/freshly_collected_data -w /opt/home/language_conditioned_irl/freshly_collected_data valaygaurang/pyrep-lgr:0.1 bash
```


## Painfull Useless Info For setup From Scratch
- Requires (`orocos_kinamtics_dynamics`)[https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/INSTALL.md#without-catkin]
- Follow To Letter : https://github.com/orocos/orocos_kinematics_dynamics/issues/115#issuecomment-363828002
- Fix in Imported Lib : https://github.com/ros/kdl_parser/issues/44#issuecomment-689672444
- also Install `sudo apt install libeigen3-dev libcppunit-dev python-sip-dev python3-sip-dev python-psutil python3-psutil python-future python3-future`
- 