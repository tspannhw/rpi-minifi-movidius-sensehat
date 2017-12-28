 #!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M")

fswebcam -r 1280x720 --no-banner /opt/demo/images/$DATE.jpg

python3 -W ignore image-classifier2.py /opt/demo/images/$DATE.jpg
