#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# ****************************************************************************
# How to classify images using DNNs on Intel Neural Compute Stick (NCS)

# Forked by Tim Spann and added Sense Hat Code and JSON
# 2017-December-28

from sense_hat import SenseHat
import json
import sys, socket
import os
import psutil
import subprocess
import time
import datetime
from time import sleep
from time import gmtime, strftime
import mvnc.mvncapi as mvnc
import skimage
from skimage import io, transform
import numpy

starttime= strftime("%Y-%m-%d %H:%M:%S",gmtime())

# User modifiable input parameters
NCAPPZOO_PATH           = os.path.expanduser( '~/workspace/ncappzoo' )
GRAPH_PATH              = NCAPPZOO_PATH + '/caffe/GoogLeNet/graph'
IMAGE_PATH              = sys.argv[1]
LABELS_FILE_PATH        = NCAPPZOO_PATH + '/data/ilsvrc12/synset_words.txt'
IMAGE_MEAN              = [ 104.00698793, 116.66876762, 122.67891434]
IMAGE_STDDEV            = 1
IMAGE_DIM               = ( 224, 224 )

# ---- Step 1: Open the enumerated device and get a handle to it -------------

# Look for enumerated NCS device(s); quit program if none found.
devices = mvnc.EnumerateDevices()
if len( devices ) == 0:
    print( 'No devices found' )
    quit()

# Get a handle to the first enumerated device and open it
device = mvnc.Device( devices[0] )
device.OpenDevice()

# ---- Step 2: Load a graph file onto the NCS device -------------------------

# Read the graph file into a buffer
with open( GRAPH_PATH, mode='rb' ) as f:
    blob = f.read()

# Load the graph buffer into the NCS
graph = device.AllocateGraph( blob )

# ---- Step 3: Offload image onto the NCS to run inference -------------------

# Read & resize image [Image size is defined during training]
img = print_img = skimage.io.imread( IMAGE_PATH )
img = skimage.transform.resize( img, IMAGE_DIM, preserve_range=True )

# Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
img = img[:, :, ::-1]

# Mean subtraction & scaling [A common technique used to center the data]
img = img.astype( numpy.float32 )
img = ( img - IMAGE_MEAN ) * IMAGE_STDDEV

# Load the image as a half-precision floating point array
graph.LoadTensor( img.astype( numpy.float16 ), 'user object' )

# ---- Step 4: Read & print inference results from the NCS -------------------

# Get the results from NCS
output, userobj = graph.GetResult()

labels = numpy.loadtxt( LABELS_FILE_PATH, str, delimiter = '\t' )

order = output.argsort()[::-1][:6]

#### Initialization

external_IP_and_port = ('198.41.0.4', 53)  # a.root-servers.net
socket_family = socket.AF_INET

host = os.uname()[1]

def getCPUtemperature():
    res = os.popen('vcgencmd measure_temp').readline()
    return(res.replace("temp=","").replace("'C\n",""))

def IP_address():
        try:
            s = socket.socket(socket_family, socket.SOCK_DGRAM)
            s.connect(external_IP_and_port)
            answer = s.getsockname()
            s.close()
            return answer[0] if answer else None
        except socket.error:
            return None

cpuTemp=int(float(getCPUtemperature()))
ipaddress = IP_address()

host = os.uname()[1]
rasp = ('armv' in os.uname()[4])
cpu = psutil.cpu_percent(interval=1)
if rasp:
    f = open('/sys/class/thermal/thermal_zone0/temp', 'r')
    l = f.readline()
    ctemp = 1.0 * float(l)/1000
usage = psutil.disk_usage("/")
mem = psutil.virtual_memory()
diskrootfree =  "{:.1f} MB".format(float(usage.free) / 1024 / 1024)
mempercent = mem.percent
external_IP_and_port = ('198.41.0.4', 53)  # a.root-servers.net
socket_family = socket.AF_INET

ipaddress = IP_address()

# Sense Hat
sense = SenseHat()
sense.clear()
temp = sense.get_temperature()
temp = round(temp, 2)
humidity = sense.get_humidity()
humidity = round(humidity, 1)
pressure = sense.get_pressure()
pressure = round(pressure, 1)
orientation = sense.get_orientation()
pitch = orientation['pitch']
roll = orientation['roll']
yaw = orientation['yaw']
acceleration = sense.get_accelerometer_raw()
x = acceleration['x']
y = acceleration['y']
z = acceleration['z']
#cputemp = out
x=round(x, 0)
y=round(y, 0)
z=round(z, 0)
pitch=round(pitch,0)
roll=round(roll,0)
yaw=round(yaw,0)


# yyyy-mm-dd hh:mm:ss
currenttime= strftime("%Y-%m-%d %H:%M:%S",gmtime())
row =  {  'cputemp2': round(ctemp,2), 'temp': temp, 'tempf': round(((temp * 1.8) + 12),2), 'humidity': humidity, 'pressure': pressure, 'pitch': pitch, 'roll': roll, 'yaw': yaw, 'x': x, 'y': y, 'z': z,'memory': mempercent, 'diskfree': diskrootfree, 'label1': labels[order[0]], 'label2': labels[order[1]], 'label3': labels[order[2]], 'label4': labels[order[3]], 'label5': labels[order[4]], 'currenttime': currenttime, 'host': host, 'cputemp': round(cpuTemp,2), 'ipaddress': ipaddress, 'starttime': starttime }
json_string = json.dumps(row)
print(json_string)


# ---- Step 5: Unload the graph and close the device -------------------------

graph.DeallocateGraph()
device.CloseDevice()

# ==== End of file ===========================================================

