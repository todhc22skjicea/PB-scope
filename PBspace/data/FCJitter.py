import torch  
import numpy as np  
import random  
from PIL import Image  
import torchvision.transforms as T  
  
class FCJitter(object):  
    def __init__(self, brightness=0.01, contrast=0.1, saturation=0.1, hue=0.1):  
        self.brightness = brightness  
        self.contrast = contrast  
        self.saturation = saturation  
        self.hue = hue     
  
    def __call__(self, img):  
            img_array = np.array(img, dtype=np.float32) / 255.0  
  
            # color jitter to every channel
            jittered_imgs = []  
            for i in range(3):  # 
                jitter = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)  
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