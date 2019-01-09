import cv2
import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


def read_captcha(path, imname):
    im = Image.open(path + imname).convert('RGB').filter(ImageFilter.MedianFilter())
    #enhancer = ImageEnhance.Contrast(im)
    #im = enhancer.enhance(2).convert('1')
    im.save(path + 'temp.jpg')
    return pytesseract.image_to_string(Image.open(path + 'temp.jpg'), config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

print read_captcha('../captchas/', 'captcha.png')






