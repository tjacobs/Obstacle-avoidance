# Obstacle avoidance
Building a neural network for avoiding obstacles on the Udacity self driving car simulator track.

## Requirements
I ran it on:

Keras 2.1.1

TensorFlow 1.4

## Simulator
https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip

https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip

https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip

## Train

To train detect_model.h5, run:

python train.py

The model.h5 is already trained to drive the track

## Run

Start the Term 1 Udacity self driving car simulator, and run:

python drive.py model.h5 data

