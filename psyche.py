#!/usr/bin/env python3
import argparse

from picamera import PiCamera

from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection
from aiy.vision.annotator import Annotator

import os, sys
import time
import subprocess, signal, psutil

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dialogflow.json"
import google.cloud.logging
client = google.cloud.logging.Client()
client.setup_logging()
import logging
def log(msg):
    logging.info("{}: {}".format(os.uname()[1], msg))

JOY_BOUNDARY_LOW = 0.2
JOY_BOUNDARY_HIGH = 0.8
FACE_PRESENCE_TIME = 4
WAIT_AFTER_SESSION = 10
MEDIA_PATH = "/home/pi/AIY-projects-python/src/examples/vision/media"
OMXPLAYER = ['omxplayer', '--no-osd']
DEVNULL = open('/dev/null', 'w')

def get_random_media_path(folder):
    try:
        media_files = os.listdir(os.path.join(MEDIA_PATH, folder))
        if len(media_files):
            return os.path.join(MEDIA_PATH, folder, random.choice(media_files))
        else:
            return False
    except OSError:
        return False

def avg_joy_score(faces):
    if faces:
        return sum(face.joy_score for face in faces) / len(faces)
    return 0.0

def elapsed_time(timer):
    return time.time() - timer

def get_joy_media(joy_score):
    if (joy_score < JOY_BOUNDARY_LOW):
        return get_random_media_path('sad')
    if (joy_score > JOY_BOUNDARY_HIGH):
        return get_random_media_path('happy')

    return get_random_media_path('average')

def kill_player(process_pid):
    try:
        process = psutil.Process(process_pid)
    except psutil.NoSuchProcess:
        return
    for pid in process.children(recursive=True):
        os.kill(pid.pid, signal.SIGTERM)
        os.kill(process_pid, signal.SIGTERM)

def play_loop():
    return subprocess.Popen(
        OMXPLAYER + ['--loop', os.path.join(MEDIA_PATH, 'loop.mp4')],
        stdin=DEVNULL,
        stdout=DEVNULL,
        stderr=DEVNULL,
        preexec_fn=os.setsid)

def kill_subprocesses_and_exit(*args):
    global omxplayer
    os.killpg(omxplayer.pid, signal.SIGTERM)
    os.system("pkill -9 -f omxplayer")
    os.system("pkill -9 -f omxplayer.bin")
    sys.exit(0)

signal.signal(signal.SIGINT, kill_subprocesses_and_exit)

log("startup")

mode = 'detect'
timer = False
joy_scores = []
omxplayer = play_loop()
with PiCamera(sensor_mode=4, resolution=(1640, 1232), framerate=30) as camera:
    with CameraInference(face_detection.model()) as inference:
        for result in inference.run(None):
            faces = face_detection.get_faces(result)
            if mode == 'detect':
                if len(faces):
                    if not timer:
                        log("start timer")
                        timer = time.time()
                else:
                    timer = False

                if timer:
                    print("face detected for seconds: %i" % elapsed_time(timer))
                else:
                    print("no face detected")

                if timer and elapsed_time(timer) > FACE_PRESENCE_TIME:
                    log("start session")
                    mode = 'session'
                    kill_player(omxplayer.pid)
                    omxplayer = subprocess.Popen(
                        OMXPLAYER + [get_random_media_path('welcome')], 
                        stdin=DEVNULL,
                        stdout=DEVNULL,
                        stderr=DEVNULL,
                        preexec_fn=os.setsid)

            if mode == 'session':
                joy_scores.append(avg_joy_score(faces))
                if omxplayer.poll() != None:
                    joy_score = sum(joy_scores) / len(joy_scores)
                    log("play %s (%f)" % (get_joy_media(joy_score), joy_score))
                    subprocess.call(
                        OMXPLAYER + [get_joy_media(joy_score)],
                        stdin=DEVNULL,
                        stdout=DEVNULL,
                        stderr=DEVNULL,
                        preexec_fn=os.setsid)
                    mode = 'detect'
                    timer = False
                    joy_scores = []
                    omxplayer = play_loop()
                    time.sleep(WAIT_AFTER_SESSION)
