#!/bin/bash
project_name="dg_analogy"
container_name=$(echo ${project_name})
docker_image="${USER}/${project_name}"

docker run --rm -it --name ${container_name} \
-u $(id -u):$(id -g) \
-v $PWD:/workspace \
-e OPENAI_API_KEY=${OPENAI_API_KEY} \
${docker_image} bash