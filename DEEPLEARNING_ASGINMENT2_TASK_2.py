#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#LADI HARISH KUMAR #20MAI0052 #Deep Learning and its Applications (CSE6037) LAB

Assignment 2 
LAB TASK 2 
Take a random image and apply convolution method using this following filters
(2A) Applying Box filter of 3 X 3 and 5 X 5 and compare it.
(2B)Applying Box filter of 3 X 3 and 5 X 5 and compare it. HERE THE STRIDE=2 
(2C)Apply zero padding before applying Box filter of 3 X 3 and 5 X 5 and compare it
And also Entropy of the input image is being calculated
# In[67]:


#importing required libraries 
from cv2 import cv2
import numpy as np
from PIL import Image


# In[78]:


#reading the image from the local drive
cat=cv2.imread("C:/cat_image.jpg")
#resizing the image square dimension
cat=cv2.resize(cat,(600,600))
#defining the filter of 3 X 3 size and applying it first
#filter1=np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9)
#defining the filter of 5 X 5 size and comapre it
filter1=np.array([(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1)])*(1/25)
print(filter1)


# In[79]:


C=cat.shape
F=filter1.shape
#converting the colouerd cat image into grayscale image or (Binary conversion)
cat_gray=cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)
cv2.imshow("Before_coverting",cat)
cv2.waitKey(0)


# In[80]:


cv2.imshow("After_coverting",cat_gray)
cv2.waitKey(0)
print(cat_gray)


# In[81]:


print(cat_gray)
print(type(cat_gray))
cat_gray2=cat_gray
print(cat_gray2)
print(type(cat_gray2))


# In[82]:


#(2A) Applying Box filter of 3 X 3 and 5 X 5 and comapre it.
import math
def pro_sum(m,n):
    sum=0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            sum=sum+(m[i,j]*n[i,j])
    return math.ceil(sum)
X1=np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9)
X2=np.array([(50,50,49),(51,50,50),(51,50,50)])
print(pro_sum(X1,X2))
for i in range(C[0]):
    for j in range(C[1]):
        k = cat_gray[i:i+F[0],j:j+F[1]]#slicing the image in the form of multiple filter dimension 
        l = pro_sum(k,filter1)
        cat_gray2[i][j]=l
print(cat_gray2)        
cv2.imshow("The convoluted image is ",cat_gray2)
cv2.waitKey(0)


# In[83]:


#(2B)Applying Box filter of 3 X 3 and 5 X 5 and comapre it.HERE THE STRIDE=2
cat_gray3=cat_gray
import math
def pro_sum(m,n):
    sum=0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            sum=sum+(m[i,j]*n[i,j])
    return math.ceil(sum)

for i in range(C[0]):
    for j in range(C[1]):
        k = cat_gray[i:i+F[0],j:j+F[1]]#slicing the image in the form of multiple filter dimension 
        l = pro_sum(k,filter1)
        cat_gray3[i][j]=l
        j+=1 #here the stride is 2 or filter jumps by 2 pixals intead of 1
    i+=1
print(cat_gray3)        
cv2.imshow("The convoluted image is ",cat_gray3)
cv2.waitKey(0)


# In[84]:


#(2C) Apply zero padding before applying Box filter of 3 X 3 and 5 X 5 and comapre it
A=C[0]+F[0]-1
B=C[1]+F[1]-1
Y= np.zeros((A,B))#creating the image of all zero intensity pixals
print(Y)
print(Y.shape)
Z=Y


# In[85]:


#fitting the input image in the centre of zero intensity pixals image Y to form zero padeded image Z
for i in range(C[0]):
    for j in range(C[1]):
        m=np.int((F[0]-1)/2)
        n=np.int((F[1]-1)/2)
        Z[i+m,j+n]=cat_gray[i,j]
print("The pixal values after zero padding is ")
print(Z)
print(Z.shape)
print(Z[0,0],Z[1,1])


# In[86]:


cat_gray4=cat_gray
import math
def pro_sum(m,n):
    sum=0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            sum=sum+(m[i,j]*n[i,j])
    return math.ceil(sum)

for i in range(C[0]):
    for j in range(C[1]):
        k = Z[i:i+F[0],j:j+F[1]]#slicing the image in the form of multiple filter dimension 
        l = pro_sum(k,filter1)
        cat_gray3[i][j]=l

print(cat_gray3)        
cv2.imshow("The convoluted image is ",cat_gray4)
cv2.waitKey(0)


# In[66]:


#Entropy of the input image
import skimage.measure    
entropy = skimage.measure.shannon_entropy(cat_gray)
print(entropy)


# In[ ]:




