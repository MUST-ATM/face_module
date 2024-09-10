#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Descri: face_recongnition
# @Author: LUXP
# Date: 2018-04-27

import face_recognition as FR
import cv2
import os
import pickle

FolderName = 'FacePerception/FaceDatabase'

## Encoding all faces in the specified folder
def EncodingFolder(FolderName):
	Faces_encodings = []
	Faces_names = []

	## get all file names in the specifiled folder
	FileList = os.listdir(FolderName)
	if '.DS_Store' in FileList:
		FileList.remove('.DS_Store')

	## encoding face files
	for file in FileList:
		FaceImage = FR.load_image_file(FolderName+'/'+file)
		FaceEncoding = FR.face_encodings(FaceImage)
		if len(FaceEncoding) == 0:
			print('Face NOT detected!!!')
			continue

		Faces_encodings.append(FaceEncoding[0])
		Faces_names.append(file[:-5])
	return Faces_names, Faces_encodings

##	Initialize Video Capture and Detech the Face, Recoginize
def DetectAndRecognize(KnownNameList, KnownEncodingList, image):
	# Initialize some variables
	face_locations = []
	face_encodings = []

	######### Face_Recognition and Matching #############
	# Find all the faces and face encodings in the current frame of video
	face_locations = FR.face_locations(image)
	face_encodings = FR.face_encodings(image, face_locations)

	for face_encode in face_encodings:
		matches = FR.compare_faces(KnownEncodingList, face_encode, 0.3)
		name = None
		if True in matches:
			first_match_index = matches.index(True)
			name = KnownNameList[first_match_index]
		return name

		#########  Mark frame and Show frame ################



def newFace(image,name):
	######### Face_Recognition and Matching #############
	# Find all the faces and face encodings in the current frame of video
	face_locations = FR.face_locations(image)
	face_encodings = FR.face_encodings(image, face_locations)
	top = face_locations[0][0]
	right = face_locations[0][1]
	bottom = face_locations[0][2]
	left = face_locations[0][3]
	NewFaceFile = FolderName + '/' + str(name) + '.png'
	cv2.imwrite(NewFaceFile, image[top:bottom, left:right])
	Faces_names, Faces_encodings = EncodingFolder(FolderName)
	print('Faces Loaded: \n', sorted(Faces_names))
	WriteIntoFile('pk_FacesName.pk', Faces_names)
	WriteIntoFile('pk_FacesEncoding.pk', Faces_encodings)

## Write encoding-faces into file
def WriteIntoFile(filename, pk_data):
	fd = open(filename, 'wb')
	pickle.dump(pk_data, fd)
	fd.close()

def ReadFromFile(filename):
	fd = open(filename, 'rb')
	pk_data = pickle.load(fd)
	fd.close()
	return pk_data

def main(img_path):
	## Encoding all faces in the specified folder
	if os.path.exists('pk_FacesName.pk'):
		Faces_names = ReadFromFile('pk_FacesName.pk')
		Faces_encodings = ReadFromFile('pk_FacesEncoding.pk')
	else:
		Faces_names, Faces_encodings = EncodingFolder(FolderName)
		print('Faces Loaded: \n', sorted(Faces_names))
		WriteIntoFile('pk_FacesName.pk', Faces_names)
		WriteIntoFile('pk_FacesEncoding.pk', Faces_encodings)
	
	image = cv2.imread(img_path)
	## Detect Faces and Recognize
	return DetectAndRecognize(Faces_names, Faces_encodings, image)

## main function
if __name__ == '__main__':
	newFace(cv2.imread('D:/PythonCode/Face-module/FacePerception/HaomingZou.jpg'), 'HaomingZou')
	print(main(img_path='D:/PythonCode/Face-module/FacePerception/HaomingZou.jpg'))