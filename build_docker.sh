#!/bin/bash
set -eu

if [[ ! -f docker_repository_name ]]; then
	echo "Please \`cp docker_repository_name.example docker_repository_name\` and edit it."
	echo "You need to prepare two repositories named 'sagemaker-rl-chainer' and"
	echo "'rovboschool_chainerrl' in your ECR. Please create in the console site."
	echo "In addition, you need to build 'chainerrl0.5.0-cpu-py3' container by yourself."
	echo "Please refer to https://gist.github.com/ikeyasu/71a4b07ce6ecc7465c4e984ac5f5c855"
	exit 2
fi
REPO=`cat docker_repository_name`

FROM_IMAGE=${REPO}/sagemaker-rl-chainer:chainerrl0.5.0-cpu-py3
IMAGE=${REPO}/roboschool_chainerrl:cpu

docker build \
	-t ${IMAGE} \
	-f Dockerfile.sagemaker_cpu \
	--build-arg FROM_IMAGE=${FROM_IMAGE} \
	.

echo "Please run \`docker push ${IMAGE}\`"
