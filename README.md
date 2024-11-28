# Face-module
[![wakatime](https://wakatime.com/badge/user/5cfddb9c-d0f0-4d5b-bf41-13cc6c3aeccd/project/ded00d1e-3852-40f9-9b8d-1d9cb720db69.svg)](https://wakatime.com/badge/user/5cfddb9c-d0f0-4d5b-bf41-13cc6c3aeccd/project/ded00d1e-3852-40f9-9b8d-1d9cb720db69)

## Description
This module is part of the project `MUST-ATM` and is responsible for the face detection and recognition. It uses the `dlib` library to detect faces and the `OpenCV` library to recognize them.

## Requirements
- Python >=3.6
- OpenCV >=3.4.2
- Numpy >=1.16.2 or <=1.26.4
- Dlib >=19.16.0
- Dassl.pytorch >=0.6.3
- scipy <=1.11.4
## Installation exclusive Face-AntiSpoofing

```bash
mamba install face_recognition
pip install opencv-python
```
## Face-AntiSpoofing based CLIP
```bash
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/
pip install -r requirements.txt
python setup.py install
pip install matplotlib

```
## Train the Face-AntiSpoofing Module
TBD...

You need a Uniattack datasets to do the train, but currently I could not provide to you.
If you really want to train the model, please contact to me.

Check the ```FaceAntiSpoofing/scripts``` folder if you already have the Uniattack datasets.

At least it works on my machine.

## About our Face-AntiSpoofing Module
The Face Anti-Spoofing module is not fully opensource according to some reason, please alternate it to other OpenSource if you need.

## Alternative Anti-Spoofing Module
The only thing that you have to change is make ```FaceAntiSpoofing.FaceAntiSpoofing.py``` could return a ```Boolean``` type when you input a image.
