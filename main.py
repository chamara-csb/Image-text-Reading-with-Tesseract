# import cv2
# import pytesseract
# from skimage.metrics import structural_similarity
# import os.path as osp
# import glob
# from skimage.transform import resize
import keras_ocr
import json
import cv2
import urllib
from PIL import Image
import requests
from io import BytesIO
import glob
import numpy as np
from skimage.transform import resize

# pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR\\tesseract.exe'
#
# img1 = cv2.imread('img2.jpg')
url = 'https://www.lankapropertyweb.com/pics/5412793/5412793_1673077832_4226.jpeg'
with urllib.request.urlopen(url) as url:
    s = url.read()
    arr = np.asarray(bytearray(s), dtype=np.uint8)
    img1 = cv2.imdecode(arr,-1)
    # img1 = cv2.imread(img1)
#
# img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#
# sinhala = pytesseract.image_to_string(img, lang='sin')
# print(sinhala)
#
# text = pytesseract.image_to_string(img)
#
# print(text)
#
# print("Text length :",len(sinhala)+len(text))
# # print(text)

def orb_sim(img1, img2):
    # SIFT is no longer available in cv2 so using ORB
    orb = cv2.ORB_create()

    # detect keypoints and descriptors
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # define the bruteforce matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # perform matches.
    matches = bf.match(desc_a, desc_b)
    # Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)



img_folder = 'images/*'

list1 = []


for path in glob.glob(img_folder):
    # print(path)
    img2 = cv2.imread(path, cv2.COLOR_BGR2RGB)
    # img3 = resize(img2, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)


    orb_sim_out = orb_sim(img1, img2)
    list1.append(orb_sim_out)

print(list1)
print(len(list1))
print("Max score is : ", max(list1)*100)

