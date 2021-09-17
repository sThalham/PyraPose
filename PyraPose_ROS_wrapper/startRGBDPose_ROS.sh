#!/bin/bash

sudo docker build --no-cache -t rgbdpose_ros .
thispid=$(sudo docker run --gpus device=0 --network=host --name=rgbdpose_ros -t -d -v ~/stefan:/stefan rgbdpose_ros)

#sudo nvidia-docker exec -it $thispid bash

#sudo nvidia-docker container kill $thispid
#sudo nvidia-docker container rm $thispid


