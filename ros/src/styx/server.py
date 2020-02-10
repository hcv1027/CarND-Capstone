#!/usr/bin/env python

# from conf import conf
# from bridge import Bridge
# from flask import Flask, render_template
# import time
# import socketio
# import eventlet.wsgi
# import eventlet
# eventlet.monkey_patch(socket=True, select=True, time=True)


# # sio = socketio.Server(async_mode='eventlet')
# sio = socketio.Server()
# app = Flask(__name__)
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

import socketio
import time

from bridge import Bridge
from conf import conf

sio = socketio.Server(async_mode='gevent')
msgs = []

dbw_enable = False
camera_counter = 0


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)


def send(topic, data):
    sio.emit(topic, data=data, skip_sid=True)


bridge = Bridge(conf, send)


@sio.on('telemetry')
def telemetry(sid, data):
    global dbw_enable
    if data["dbw_enable"] != dbw_enable:
        dbw_enable = data["dbw_enable"]
        bridge.publish_dbw_status(dbw_enable)
    bridge.publish_odometry(data)


@sio.on('control')
def control(sid, data):
    bridge.publish_controls(data)


@sio.on('obstacle')
def obstacle(sid, data):
    bridge.publish_obstacles(data)


@sio.on('lidar')
def obstacle(sid, data):
    bridge.publish_lidar(data)


@sio.on('trafficlights')
def trafficlights(sid, data):
    bridge.publish_traffic(data)


@sio.on('image')
def image(sid, data):
    global camera_counter
    if camera_counter % 3 == 0:
        camera_counter = 0
        bridge.publish_camera(data)
    camera_counter += 1


if __name__ == '__main__':

    # Create socketio WSGI application
    app = socketio.WSGIApp(sio)

    # deploy as an gevent WSGI server
    pywsgi.WSGIServer(
        ('', 4567), app, handler_class=WebSocketHandler).serve_forever()
