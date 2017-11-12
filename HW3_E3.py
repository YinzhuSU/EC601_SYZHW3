import cv2
import random
import numpy as np
def Add_gaussian_Noise(src,mean,sigma):
    noiseArr = src.copy()
    noiseArr = np.random.normal(mean,sigma,src.shape)
    np.add(src,noiseArr,src,casting="unsafe")
    return;
def Add_salt_pepper_Noise(img,pa,pb):
    row,col,ch=img.shape
    amount1=row*col*pa
    amount2=row*col*pb
    for i in range(int(amount1)):
        img[int(random.uniform(0,row))][int(random.uniform(0,col))]=0
    for i in range(int(amount2)):
        img[int(random.uniform(0,row))][int(random.uniform(0,col))]=255
        
    
img=cv2.imread('cameraman.png')
cv2.namedWindow('Original image')
cv2.imshow('Original',img)
cv2.imwrite('Original.jpg',img)

noise_img=img.copy()
mean=0
sigma=50
Add_gaussian_Noise(noise_img,mean,sigma)
cv2.imshow('Gaussian Noise',noise_img)
cv2.imwrite('GaussianNoise.jpg',noise_img)

noise_dst=noise_img.copy()
cv2.blur(noise_dst,(3,3))
cv2.imshow('Box filter',noise_dst)
cv2.imwrite('Boxfilter.jpg',noise_dst)

noise_dst1=noise_img.copy()
cv2.GaussianBlur(noise_dst1,(3,3),1.5)
cv2.imshow('GaussianBlur filter',noise_dst1)
cv2.imwrite('GaussianBlurfilter.jpg',noise_dst1)

noise_dst2=noise_img.copy()
cv2.medianBlur(noise_dst2,3)
cv2.imshow('Median filter',noise_dst2)
cv2.imwrite('Medianfilter.jpg',noise_dst2)

noise_img2=img.copy()
pa=0.01
pb=0.01
Add_salt_pepper_Noise(noise_img2,pa,pb)
cv2.imshow("Salt and Peper Noise", noise_img2)
cv2.imwrite("SaltandPeperNoise.jpg", noise_img2)

noise_dst3=noise_img2.copy()
cv2.blur(noise_dst3,(3,3))
cv2.imshow('Box filter2',noise_dst3)
cv2.imwrite('Boxfilter2.jpg',noise_dst3)

noise_dst4=noise_img2.copy()
cv2.GaussianBlur(noise_dst4,(3,3),1.5)
cv2.imshow('GaussianBlur filter2',noise_dst4)
cv2.imwrite('GaussianBlurfilter2.jpg',noise_dst4)

noise_dst5=noise_img2.copy()
cv2.medianBlur(noise_dst5,3)
cv2.imshow('Medianfilter2',noise_dst5)
cv2.imwrite('Medianfilter2.jpg',noise_dst5)