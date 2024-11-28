from FaceRecognition import faceRecognitionByPath, faceRecognitionByByte
import cv2
import numpy as np


def test_face_recognition_by_path():
    image_path = "C:/misc/API/backend/capture2.jpg"  
    result = faceRecognitionByPath(image_path)
    print(f"Recognition result by path: {result}")

def test_face_recognition_by_byte():

    image = cv2.imread("C:/misc/API/backend/capture2.jpg")
    _, byte_image = cv2.imencode('.jpg', image)
    
    byte_image = np.frombuffer(byte_image, np.uint8)  
    
    result = faceRecognitionByByte(byte_image)
    print(f"Recognition result by byte: {result}")

if __name__ == "__main__":
    test_face_recognition_by_path()
    test_face_recognition_by_byte()