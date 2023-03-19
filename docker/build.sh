#!/usr/bin/env bash

SSH_KEY=$(cat ~/.ssh/id_rsa)
IMG_TAG=${1:-"agod:torch1.13.1-cu113"}

echo "Building ${IMG_TAG} ..."
sudo docker build --build-arg SSH_KEY="${SSH_KEY}" -f Dockerfile -t ${IMG_TAG} ..

sudo docker run -dit --gpus all --name agod -v ${HOME}/agod:/root ${IMG_TAG}
echo "Run image ${IMG_TAG} as container agod"
echo "Run 'sudo docker exec -ti agod bash' to enter the container"
