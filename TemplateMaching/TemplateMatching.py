import numpy as np
import cv2

def TemplateMatching(src, temp, stepsize): # src: source image, temp: template image, stepsize: the step size for sliding the template
    mean_t = 0;
    var_t = 0;
    location = [0, 0];
    # Calculate the mean and variance of template pixel values
    # ------------------ Put your code below ------------------ 
    for i in range(0,temp.shape[0]): #0 means row
        for j in range(0, temp.shape[1]): #1 means column
            mean_t = temp[i,j] + mean_t
    mean_t = mean_t/(temp.shape[0] * temp.shape[1]) 

    for i in range(0,temp.shape[0]): #0 means row
        for j in range(0, temp.shape[1]): #1 means column        
            var_t = (temp[i,j] - mean_t)**2 + var_t
    var_t = var_t/(temp.shape[0] * temp.shape[1])
    
    max_corr = 0;
    # Slide window in source image and find the maximum correlation
    for i in np.arange(0, src.shape[0] - temp.shape[0], stepsize):
        for j in np.arange(0, src.shape[1] - temp.shape[1], stepsize):
            mean_s = 0;
            var_s = 0;
            corr = 0;
            
            # Calculate the mean and variance of source image pixel values inside window
            # ------------------ Put your code below ------------------ 
            for a in range(0,temp.shape[0]): #0 means row
                for b in range(0, temp.shape[1]): #1 means column
                    mean_s = src[a+i,b+j] + mean_s
            mean_s = mean_s/(src.shape[0] * src.shape[1]) 

            for a in range(0,temp.shape[0]): #0 means row
                for b in range(0, temp.shape[1]): #1 means column        
                    var_s = (src[a+i,b+j] - mean_s)**2 + var_s
            var_s = var_s/(src.shape[0] * src.shape[1])            
            
            
            
            # Calculate normalized correlation coefficient (NCC) between source and template
            # ------------------ Put your code below ------------------ 
            for a in range(0,temp.shape[0]): #0 means row
                for b in range(0, temp.shape[1]): #1 means column
                    corr = ((temp[a,b]-mean_t) * (src[a,b]-mean_s)) + corr
            corr = corr /(var_t * var_s * temp.size)
                        
            if corr > max_corr:
                max_corr = corr;
                location = [i, j];
    return location

# load source and template images
source_img = cv2.imread('/Users/suyinzhu/Desktop/EC601/OpenCV/source_img.jpg',0) # read image in grayscale
temp = cv2.imread('/Users/suyinzhu/Desktop/EC601/OpenCV/template_img.jpg',0) # read image in grayscale
location = TemplateMatching(source_img, temp, 20); #20
print(location)
match_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB)

# Draw a red rectangle on match_img to show the template matching result
# ------------------ Put your code below ------------------
(i,j) = location
#(h,w) = temp.shape 
cv2.rectangle(match_img, (i,j), (i+temp.shape[1], j+temp.shape[0]), (0,0,255), 2)

# Save the template matching result image (match_img)
# ------------------ Put your code below ------------------ 
cv2.imwrite('Result.png', match_img)

# Display the template image and the matching result
cv2.namedWindow('TemplateImage', cv2.WINDOW_NORMAL)
cv2.namedWindow('MyTemplateMatching', cv2.WINDOW_NORMAL)
cv2.imshow('TemplateImage', temp)
cv2.imshow('MyTemplateMatching', match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()