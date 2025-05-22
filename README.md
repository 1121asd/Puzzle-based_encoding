# Z-Encoding & Sparse LSTM for Traffic Shockwave diagram

An anonymous implementation of a modular encoding method (“Z-encoding”) and a corresponding sparse LSTM architecture for learning traffic‐shockwave patterns.

## Description

This repository contains:
loopdetectordata folder includes loop detector data downloaded from https://catalog.data.gov/dataset/seattle-20-second-freeway and processed as volume and density, and game schedule data. 

Puzzle/Pixel based encoding generation file: generate shockwave diagrams from loop detector data and then encoded as LSTM input using puzzle/pixel based scheme. 

Z_LSTM file: feeds the puzzle-based encoding results to vanilla LSTM to train and test.
X_9by12 file: feeds the pixel-based (9by12 resolution) encoding results to vanilla LSTM to train and test.
Sparse_LSTM file: Sparse_LSTM module to replace vanilla LSTM

Plot heatmap and violin plot, oneday_average day performance visualization files: performance visualization. 

This code is released under the MIT License.
