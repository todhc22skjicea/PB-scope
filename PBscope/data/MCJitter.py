import torch  
import numpy as np  
import random  
from PIL import Image  
import torchvision.transforms as T  

'''
MCjitter

'''
class MCJitter(object):  
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):  
        self.brightness = brightness  
        self.contrast = contrast  
        self.saturation = saturation       
    def __call__(self, img):  
            img_array = np.array(img, dtype=np.float32) / 255.0  

            # color jitter to every channel
            jittered_imgs = []  
            for i in range(3):  # 
                jitter = T.RandomApply([T.ColorJitter(self.brightness, self.contrast, self.saturation)],p=0.8) 
                jittered_img = jitter(Image.fromarray((img_array[..., i] * 255).astype(np.uint8)))  
                jittered_imgs.append(np.array(jittered_img, dtype=np.float32) / 255.0)  
  
            jittered_array = np.stack(jittered_imgs, axis=-1)  
            result_img = Image.fromarray((jittered_array * 255).astype(np.uint8))  
  
            return result_img  



       
'''  
# Example usage:  
transform = T.Compose([  
    FCJitter(0.2, 0.3, 0.4, p=0.8),  
    T.ToTensor()  
])  
  
# Apply the transform to an image  
img = Image.open('path_to_image.jpg')  
transformed_img = transform(img)
'''