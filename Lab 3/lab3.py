# %%
#LOAD IMAGE

from matplotlib import gridspec
from PIL import Image 
import cv2
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

filepath = 'image1.jpg'

imageObj = cv2.imread(filepath)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(imageObj, cv2.COLOR_BGR2RGB))
plt.savefig('1.jpg')
plt.show()



# %%
#EROSION

#img=cv2.cvtColor(imageObj,cv2.COLOR_BGR2RGB)
img=cv2.cvtColor(imageObj,cv2.COLOR_RGB2GRAY)

val_o,img_bin=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blue_color = cv2.calcHist([imageObj], [0], None, [256], [0, 256])
red_color = cv2.calcHist([imageObj], [1], None, [256], [0, 256])
green_color = cv2.calcHist([imageObj], [2], None, [256], [0, 256])

plt.title('Histogram')
plt.plot(blue_color)
plt.plot(red_color)
plt.plot(green_color)
colors =('r','g','b')
plt.savefig('2.jpg')
plt.show()



kernel=np.ones((5,5),np.float32)
img_eroded=cv2.erode(img_bin,kernel,iterations=1)
plt.title('Eroded Color Image')
plt.imshow(img_eroded)
plt.savefig('3.jpg')
plt.show()

hist_grayscale = cv2.calcHist([img], [0], None, [256], [0,256])
hist_otsu = cv2.calcHist([img_bin], [0], None, [256], [0,256])
hist_eroded = cv2.calcHist([img_eroded], [0], None, [256], [0,256])


# %%
#EROSION PLOTTING

row=2
col=3

fig=plt.figure(figsize=(15,15))
gs=GridSpec(row,col)

fig.add_subplot(gs[0,0])
plt.title('Original Grayscale Image')
plt.imshow(img,cmap='gray')

fig.add_subplot(gs[0,1])
plt.title('OTSU Binarized Image')
plt.imshow(img_bin,cmap='gray')

fig.add_subplot(gs[0,2])
plt.title('Eroded Binarized Image with 5x5 Kernel')
plt.imshow(img_eroded,cmap='gray')

fig.add_subplot(gs[1,0])
plt.title('Histogram')
plt.plot(hist_grayscale)

fig.add_subplot(gs[1,1])
plt.title('Histogram')
plt.plot(hist_otsu)

fig.add_subplot(gs[1,2])
plt.title('Histogram')
plt.plot(hist_eroded)

plt.savefig('4.jpg')

# %%
#DILATION

kernel=np.ones((5,5),np.float32)
img_dilated=cv2.dilate(img_bin,kernel,iterations=1)
plt.title('Dilated Color Image')
plt.imshow(img_dilated)
plt.savefig('5.jpg')
plt.show()

hist_dilated = cv2.calcHist([img_dilated], [0], None, [256], [0,256])

# %%
#DILATION PLOTTING

row=2
col=3

fig=plt.figure(figsize=(15,15))
gs=GridSpec(row,col)

fig.add_subplot(gs[0,0])
plt.title('Original Grayscale Image')
plt.imshow(img,cmap='gray')

fig.add_subplot(gs[0,1])
plt.title('OTSU Binarized Image')
plt.imshow(img_bin,cmap='gray')

fig.add_subplot(gs[0,2])
plt.title('Dilated Binarized Image with 5x5 Kernel')
plt.imshow(img_dilated,cmap='gray')

fig.add_subplot(gs[1,0])
plt.title('Histogram')
plt.plot(hist_grayscale)

fig.add_subplot(gs[1,1])
plt.title('Histogram')
plt.plot(hist_otsu)

fig.add_subplot(gs[1,2])
plt.title('Histogram')
plt.plot(hist_eroded)

plt.savefig('6.jpg')

# %%
#GRADIENT
kernel=np.ones((5,5),np.float32)
img_gradient = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel)
plt.title('Gradient Color Image')
plt.imshow(img_gradient)
plt.savefig('7.jpg')
plt.show()

hist_gradient = cv2.calcHist([img_gradient], [0], None, [256], [0,256])



# %%
#GRADIENT PLOTTING

row=2
col=3

fig=plt.figure(figsize=(15,15))
gs=GridSpec(row,col)

fig.add_subplot(gs[0,0])
plt.title('Original Grayscale Image')
plt.imshow(img,cmap='gray')

fig.add_subplot(gs[0,1])
plt.title('OTSU Binarized Image')
plt.imshow(img_bin,cmap='gray')

fig.add_subplot(gs[0,2])
plt.title('Gradient Binarized Image with 5x5 Kernel')
plt.imshow(img_gradient,cmap='gray')

fig.add_subplot(gs[1,0])
plt.title('Histogram')
plt.plot(hist_grayscale)

fig.add_subplot(gs[1,1])
plt.title('Histogram')
plt.plot(hist_otsu)

fig.add_subplot(gs[1,2])
plt.title('Histogram')
plt.plot(hist_gradient)

plt.savefig('8.jpg')