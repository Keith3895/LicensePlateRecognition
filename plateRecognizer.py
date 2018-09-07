import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
import pytesseract

import argparse
import io
import re

from google.cloud import storage
from google.cloud import vision
from google.protobuf import json_format


import requests
foundPlate=False

class plateRecognizer():
    def __init__(self):
        print('Dummy text')
        self.cap = cv2.VideoCapture('./output6.avi')
        self.ret, self.frame = self.cap.read()

    def recognizer(self,img): 
        threshold_img = self.preprocess(img)
        contours= self.extract_contours(threshold_img)
        # cv2.imshow('test1',img)
        callback = self.cleanAndRead(img,contours)
        if callback:
            NumberPlate,numImage = callback
            if(isinstance(NumberPlate,str)):
                if(len(NumberPlate)>6):
                    cv2.imwrite('recognized.jpg',numImage)
                    return NumberPlate

    def videoRecognition(self):
        while(True):
            self.ret, self.frame = self.cap.read()
            NumberPlate = self.recognizer(self.frame)
            cv2.imshow('frame',self.frame)
            if NumberPlate:
                return NumberPlate
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return 0
        

    def get_frame(self):
        self.frames = open("./stream.jpg", 'w+')
        ret, img = self.cap.read()
        if ret:	
            cv2.imwrite("./stream.jpg", img)
        return self.frames.read()

    def preprocess(self,img):
        imgBlurred = cv2.GaussianBlur(img, (5,5), 0)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
        ret2,threshold_img = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return threshold_img

    def cleanPlate(self,plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        im1,contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)

            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x,y,w,h = cv2.boundingRect(max_cnt)
            # print("-------------------------------------------------------------------")
            if not self.ratioCheck(max_cntArea,w,h):
                return plate,None

            cleaned_final = thresh[y:y+h, x:x+w]
            # cv2.imshow("Function Test",cleaned_final)
            return cleaned_final,[x,y,w,h]

        else:
            return plate,None


    def extract_contours(self,threshold_img):
        
        element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
        morph_img_threshold = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
        # cv2.imshow("Morphed",morph_img_threshold)
        # cv2.waitKey(0)

        im2,contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
        return contours


    def detect_text(self,path):
        """Detects text in the file."""
        client = vision.ImageAnnotatorClient()

        # [START vision_python_migration_text_detection]
        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.types.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        for text in texts:
            print('\n"{}"'.format(text.description))
            return format(text.description)

            # vertices = (['({},{})'.format(vertex.x, vertex.y)
            #             for vertex in text.bounding_poly.vertices])

            # print('bounds: {}'.format(','.join(vertices)))
        # [END vision_python_migration_text_detection]
    # [END vision_text_detection]





    def ratioCheck(self,area, width, height):
        ratio = float(width) / float(height)
        # print("ratio:",ratio)
        if ratio < 1:
            ratio = 1 / ratio
        # print("ratio:",ratio)
        aspect = 4.7272
        min = 15*aspect*15  # minimum area
        max = 125*aspect*125  # maximum area
        # print("min:",min)
        # print("max:",max)
        rmin = 3
        rmax = 6
        # print("area:",area)
        if (area < min or area > max) or (ratio < rmin or ratio > rmax):
            return False
        return True

    def isMaxWhite(self,plate):
        avg = np.mean(plate)
        # print("avg",avg)
        if(avg>=100):
            return True
        else:
            return False

    def validateRotationAndRatio(self,rect,i):
        (x, y), (width, height), rect_angle = rect
        if(width>height):
            angle = -rect_angle
        else:
            angle = 90 + rect_angle
        # print("angle",angle)

        # if angle>15:
        #  	return False

        if height == 0 or width == 0:
            return False

        area = height*width
        if not self.ratioCheck(area,width,height):
            return False
        else:
            return True

    def saveFile(self,img):
        cv2.imwrite('NumberPlate.jpg',img)
        return "./NumberPlate.jpg"

    def cleanAndRead(self,img,contours):
        count=0
        for i,cnt in enumerate(contours):
            min_rect = cv2.minAreaRect(cnt)
            x2,y2,w2,h2 = cv2.boundingRect(cnt)
            testImag = img
            cv2.rectangle(testImag,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)
            # cv2.imshow("rectsFirst",testImag)
            if self.validateRotationAndRatio(min_rect,i):

                x,y,w,h = cv2.boundingRect(cnt)
                plate_img = img[y:y+h,x:x+w]

                # print("check:",isMaxWhite(plate_img))
                if(self.isMaxWhite(plate_img)):
                                    

                    count+=1
                    clean_plate, rect = self.cleanPlate(plate_img)

                    if rect:
                        
                        x1,y1,w1,h1 = rect
                        x,y,w,h = x+x1,y+y1,w1,h1

                        # cv2.imshow("Cleaned Plate",clean_plate)
                        # cv2.waitKey(0)
                        plate_im = Image.fromarray(clean_plate)
                        text = pytesseract.image_to_string(plate_im, lang='eng')
                        print ("Detected Text : ",text)
                        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        # cv2.imshow("Detected Plate",img)
                        snip = img[y:y+h,x:x+w]
                        foundPlate=True
                        snipText =  pytesseract.image_to_string(snip, lang='eng')
                        print("test convert: ",snipText )
                        print(len(snipText))
                        if len(text)>0 or len(snipText)>0:
                            # cv2.imshow("rects",plate_img)
                            # cv2.imshow("rects1",snip)
                            return snipText,snip
                            # link = saveFile(plate_img)
                            # return detect_text(link)
                        # cv2.waitKey(0)
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera disabled and all output windows closed...")
        return()		
      
def main():
    plate = plateRecognizer()
    print(plate.videoRecognition())
    

if __name__ == '__main__':
    print ("DETECTING PLATE . . .")
    main()	