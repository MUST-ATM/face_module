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
## Checkpoint

 **Rn50**: [OnDrive](https://1drv.ms/u/c/3e4ad39ec12f7a33/Ea23JGPiYbBGlQ26Xv2JQzMByGAq5QHF1t-9ubWvKkP7Cg?e=e8dVIf)
 **Vit-16**:[OnDrive](https://studentmust-my.sharepoint.com/:u:/g/personal/1220026920_student_must_edu_mo/EbznKqw0jzZLoCtZL3xC1nsBQGGWIoF0yhoiu06AK2JbaA?e=Is1pTE)
## Train the Face-AntiSpoofing Module
You have to prepare a **Uniattack datasets** to train this model, but here is not part of Unittack work, also we do not have a direct relationship. So I could not provide this dataset to you. 
If you want to train the model, please contact me.

Check the ```FaceAntiSpoofing/scripts``` folder if you already have the ```Uniattack``` datasets.

At least it works on my machine.

## About our Face-AntiSpoofing Module
The Face Anti-Spoofing module is not fully opensource according to some reason, please alternate it to other OpenSource if you need.

## Alternative Anti-Spoofing Module
The only thing that you have to change is make ```FaceAntiSpoofing.FaceAntiSpoofing.py``` could return a ```Boolean``` type when you input a image.
