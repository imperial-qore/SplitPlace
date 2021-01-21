#!/bin/sh

docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
sudo rm -rf ~/container_data/*
