# %%
from PIL import Image
import cv2
from matplotlib import pyplot as plt


filepath = 'image1.jpg'
img = cv2.imread(filepath)
print (type(img))
print (img.shape)

# %%
plt.imshow(img)
plt.xlabel('BGR FORMAT IMAGE')
plt.savefig('BGR image.jpg')

plt.show()


# %%
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128,128), cv2.INTER_LINEAR)

plt.imshow(img)
plt.xlabel("RGB FORMAT IMAGE, RESIZED")
plt.savefig('RGB Resized image.jpg')
plt.show()


print(img.size)
print(type(img))

# %%
import numpy as np
img_data = np.array(img)
print(img_data.shape)
print(type(img_data))

new_img = Image.fromarray(img_data)
print(type(new_img))
new_img.save(filepath + '.jpg')
