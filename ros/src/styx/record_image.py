#!/usr/bin/env python

import os
import numpy as np
from PIL import Image as PIL_Image
from io import BytesIO
import base64
from flask import Flask, render_template
import time
import socketio
import eventlet.wsgi
import eventlet
eventlet.monkey_patch(socket=True, select=True, time=True)


# sio = socketio.Server(async_mode='eventlet')
sio = socketio.Server()
app = Flask(__name__)

image_counter = 0
image_idx = 0


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)


@sio.on('image')
def image(sid, data):
    # print("receive image")
    global image_idx
    global image_counter
    cwd = os.path.dirname(os.path.realpath(__file__))
    if (image_counter % 5) == 0:
        imgString = data["image"]
        image = PIL_Image.open(BytesIO(base64.b64decode(imgString)))
        image.save(cwd + '/data/' + str(image_idx) + '.png')
        image_idx += 1
    image_counter += 1
    # print("save image")


if __name__ == '__main__':

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
