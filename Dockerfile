FROM ikeyasu/reinforcement-learning:cpu
MAINTAINER ikeyasu <ikeyasu@gmail.com>

ENV DEBIAN_FRONTEND oninteractive

COPY sagemaker/train /usr/local/bin/train
COPY ./ /opt/roboschool_chainerrl

ENV APP "lxterminal -e bash"
