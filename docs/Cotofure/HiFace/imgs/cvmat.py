import sys
sys.path=sys.path + ["/Library/Python/3.7/site-packages"]

import lldb
import errno
import time
import struct
import tempfile
from subprocess import call
from time import strftime
import os
import shlex
import argparse
from os.path import expanduser
import numpy as np
try:
    import cv2
except ImportError as e:
    print("Import cv2 module error : {}".format(e))
    print("imshow module will not work")

iw_visualizer_cmd = """
if len(sys.argv) < 2:
    print("Not enough arguments")
    sys.exit()
else:
    IMG_NAME = str(sys.argv[1])
img = cv2.imread(IMG_NAME)
size = img.shape
if size[0] * size[1] > (1500*1500):
    print("Image too big")
    sys.exit(0)
if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.namedWindow('Visualizer')
cv2.imshow('Visualizer', img)
cv2.waitKey(0)
cv2.destroyWindow('Visualizer')
"""

##################################################
# __lldb_init_module ()
##################################################


def __lldb_init_module(debugger, internal_dict):
    # Initialization code to add your commands
    debugger.HandleCommand('command script add -f cvmat.imshow imshow')
    debugger.HandleCommand('command script add -f cvmat.imwrite imwrite')
    debugger.HandleCommand('command script add -f cvmat.printMat printMat')
    print('The "imshow, imwrite, printMat" command has been installed')


def imwrite(debugger, command, result, internal_dict):
    # Get the frame.
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetSelectedThread()
    frame = thread.GetFrameAtIndex(0)
    # command holds the argument passed to im_show(),
    # e.g., the name of the Mat to be displayed.
    imageName = command
    # Get access to the required memory member.
    # It is wrapped in a SBValue object.
    root = frame.FindVariable(imageName)
    # Get cvMat attributes.
    matInfo = getMatInfo(root, command)
    # Print cvMat attributes.
    printMatInfo(matInfo)
    mat = getMat(debugger, root, command)
    # Show the image.
    showImage(mat, matInfo, True)

def imshow(debugger, command, result, internal_dict):
    # Get the frame.
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetSelectedThread()
    frame = thread.GetFrameAtIndex(0)
    # command holds the argument passed to im_show(),
    # e.g., the name of the Mat to be displayed.
    imageName = command
    # Get access to the required memory member.
    # It is wrapped in a SBValue object.
    root = frame.FindVariable(imageName)
    # Get cvMat attributes.
    matInfo = getMatInfo(root, command)
    # Print cvMat attributes.
    printMatInfo(matInfo)
    mat = getMat(debugger, root, command)
    # Show the image.
    showImage(mat, matInfo)


def printMat(debugger, command, result, internal_dict):
    # Get the frame.
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetSelectedThread()
    frame = thread.GetFrameAtIndex(0)

    # command holds the argument passed to im_show(),
    # e.g., the name of the Mat to be displayed.
    imageName = command
    # Get access to the required memory member.
    # It is wrapped in a SBValue object.
    root = frame.FindVariable(imageName)
    # Get cvMat attributes.
    matInfo = getMatInfo(root, command)
    # Print cvMat attributes.
    printMatInfo(matInfo)
    mat = getMat(debugger, root, command)
    print(np.array_str(mat, precision=3, suppress_small=True))


def getMatInfo(root, command):
    # Flags.
    flags = int(root.GetChildMemberWithName("flags").GetValue())
    # Channels.
    channels = 1 + (flags >> 3) & 63

    # Type of cvMat.
    depth = flags & 7
    if depth == 0:
        cv_type_name = 'CV_8U'
        data_symbol = 'B'
    elif depth == 1:
        cv_type_name = 'CV_8S'
        data_symbol = 'b'
    elif depth == 2:
        cv_type_name = 'CV_16U'
        data_symbol = 'H'
    elif depth == 3:
        cv_type_name = 'CV_16S'
        data_symbol = 'h'
    elif depth == 4:
        cv_type_name = 'CV_32S'
        data_symbol = 'i'
    elif depth == 5:
        cv_type_name = 'CV_32F'
        data_symbol = 'f'
    elif depth == 6:
        cv_type_name = 'CV_64F'
        data_symbol = 'd'
    else:
        print("cvMat Type not sypported")

    # Rows and columns.
    rows = int(root.GetChildMemberWithName("rows").GetValue())
    cols = int(root.GetChildMemberWithName("cols").GetValue())
    # Get the step (access to value of a buffer with GetUnsignedInt16()).
    error = lldb.SBError()
    line_step = root.GetChildMemberWithName("step").GetChildMemberWithName(
        'buf').GetData().GetUnsignedInt16(error, 0)
    # Get data address.
    data_address = int(root.GetChildMemberWithName("data").GetValue(), 16)
    # Create a dictionary for the output.
    matInfo = {
        'cols': cols,
        'rows': rows,
        'channels': channels,
        'line_step': line_step,
        'data_address': data_address,
        'data_symbol': data_symbol,
        'flags': flags,
        'cv_type_name': cv_type_name,
        'name': command
    }
    return matInfo


def printMatInfo(matInfo):
    print ("flags: " + str(matInfo['flags']))
    print ("type: " + matInfo['cv_type_name'])
    print ("channels: " + str(matInfo['channels']))
    print ("rows: " + str(matInfo['rows']) + ", cols: " + str(matInfo['cols']))
    print ("line step: " + str(matInfo['line_step']))
    print ("data address: " + str(hex(matInfo['data_address'])))


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def getMat(debugger, root, command):
    matInfo = getMatInfo(root, command)

    width = matInfo['cols']
    height = matInfo['rows']
    n_channel = matInfo['channels']
    line_step = matInfo['line_step']
    data_address = matInfo['data_address']

    if width == 0 | height == 0:
        return np.array([])

    # Get the process info.
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    # Read the memory location of the data of the Mat.
    error = lldb.SBError()
    memory_data = process.ReadMemory(data_address, line_step * height, error)

    # Calculate the memory padding to change to the next image line.
    # Either due to memory alignment or a ROI.
    if matInfo['data_symbol'] in ('b', 'B'):
        elem_size = 1
    elif matInfo['data_symbol'] in ('h', 'H'):
        elem_size = 2
    elif matInfo['data_symbol'] in ('i', 'f'):
        elem_size = 4
    elif matInfo['data_symbol'] == 'd':
        elem_size = 8
    padding = line_step - width * n_channel * elem_size

    # Format memory data to load into the image.
    image_data = []
    fmt = '%d%s%dx' % (width * n_channel, matInfo['data_symbol'], padding)
    for line in chunker(memory_data, line_step):
        image_data.append(struct.unpack(fmt, line))

    arr = np.array(image_data)
    if n_channel > 1:
        arr = np.reshape(arr, (height, -1, n_channel))
    return arr


def showImage(arr, matInfo, saveOnly=False):
    imageDirectory = '/tmp/xcode_debug_images/'

    imageName = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()) + ".png"
    imagePath = imageDirectory + imageName

    try:
      os.makedirs(imageDirectory)
    except OSError as e:
      if e.errno == errno.EEXIST and os.path.isdir(imageDirectory):
          pass
      else:
          raise

    print("write to temp file " + imagePath)
    cv2.imwrite(imagePath, arr)
    if not saveOnly:
        cmd = "open {}".format(imagePath)
        print("Run : {}".format(cmd))
        os.system(cmd)
