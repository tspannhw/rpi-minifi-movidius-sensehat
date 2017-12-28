#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M")

fswebcam -q -r 1280x720 --no-banner /opt/demo/images/$DATE.jpg

python3 -W ignore ~/workspace/ncappzoo/apps/image-classifier/image-classifier2.py /opt/demo/images/$DATE.jpg
