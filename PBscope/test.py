from utils.logger import Logger
from utils.misc import export_fn
import torch
from torch.cuda.amp import autocast, GradScaler
import os
import torchvision.transforms as transforms 
from engine.criterion import clustering_accuracy_metrics
import numpy as np 
import random  
from PIL import Image 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "/home/drug_model/result_T6"
base_path = "/home/drugdataset/dataset_t6/test"
print(f"Evaluating")
model = torch.load("/home/drug_model/model/T6/CC/model.pth")
model.eval()
folders_class = ["T1648", "T1297", "T0437", "T0374", "T0429", "T0373", "T0461", "T0433", "T0364", "T0038","T0030", "T0129", "T0147", "T0163", "T0256", "T0304", "T0314", "T0327", "T0342", "T0492","T0498", "T0520", "T0263", "T0374L", "T2215", "T0973", "T1056", "T1090", "T1144","T1146", "T1158", "T1159", "T0239", "T1291", "T2205", "T1181", "T1188", "T1210", "T1222","T0167", "T2597", "T0610", "T0646", "T2565", "T0711", "T0704", "T0692", "T0679", "T0773","T0678", "T0449", "T2508", "T0445", "T1636", "T2586","T0772", "T0740", "T0875", "T0891", "T2995", "T2827", "T2858", "T0800", "T0801", "T0809", "T0858", "T0860", "T2546", "T0392", "T0928", "T0933", "T1410", "T1418", "T1431", "T1439", "T1448", "T1452", "T1454", "T1477", "T1524", "T1537", "T1546", "T1558", "T1563", "T1661", "T1737", "T2175", "T2144", "T2399", "T1630", "T1621", "T1639", "T1642", "T1659", "T1660", "T1684", "T2381", "T2364", "T2382", "T2372", "T2369", "T2145", "T2115", "T2587", "T2532", "T1835", "T0335", "T8222", "T1085L", "T0152", "T2303", "T2534", "T0097L", "T2148", "T2490", "T2483", "T3060", "T3059", "T1995", "T2325", "T2328", "T2920", "T2984", "T2851", "T6218", "T6230", "T3091", "T6227", "T2946", "T1656", "T1266", "T0080", "T0078", "T1038", "T1670", "T1829", "T1743", "T1777", "T1797", "T1912", "T1784", "T1785", "T1791", "T1792", "T1506", "T2220", "T0093L", "T2677", "T2456", "T2509", "T2539", "T2500", "T2066", "T2125", "T2485", "T2397", "T3067", "T2609", "T2656", "T1963", "T1921", "T6199", "T8387", "T14998", "T6460", "T1260", "T8151", "T1929", "T1936", "T1975", "T1894", "T1903", "T1988", "T3061", "T3211", "T6321", "T3269", "T3616", "T3402", "T3380", "T3626", "T3625", "T6115", "T3623", "T3634", "T3678", "T6165", "T6019", "T3726", "T6867", "T6121", "T6120", "T6280", "T6302", "T6758", "T6345", "T2S0007", "T6169", "T2P2923", "T6723", "T6588", "T4168", "T6156", "T6020", "T4332", "T7503", "T6880", "T6101", "T4575", "T4409", "T6487", "T6674", "T4749", "T4976", "T5001", "T5030", "T5109", "T3603", "T1633", "T0745", "T2796", "T7175", "T3O2749", "T5171", "T5177", "T5462", "T5882", "T7094", "T6930", "T5995", "T7584", "T7394", "T7486", "T7861", "T8132", "T5857", "T7604", "T8402", "T6914", "T8399", "T6475", "TQ0277", "T1791L", "T8482", "T12401", "TQ0210", "T8474", "TQ0319", "T8541", "T15732", "T8651", "T8654", "T2147", "T12317", "T8684", "T8825", "T22235", "TQ0064", "T0979", "T10358", "T19965", "T12311", "T1756L", "T22324", "T10585", "T20029", "T12594", "T15675", "T0247", "T1035", "T0878", "T2854", "T5016", "T4883", "T0194", "T2211", "T0033", "T13202","MG132", "TG", "DMSO","NA"] #Group_all

class_number = len(folders_class)
#print(class_number)
point_number = 30
cluster_labels = []
ground_truth_labels = []
ground_truth_labels_select = []
cluster_labels_select = []
feature_extract = [] 
transform = transforms.Compose([  
    transforms.Resize((224, 224)),  
    transforms.ToTensor()            
])            
def get_images_from_folder(folder_path, max_images):  
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path)] 

    random.shuffle(images)  
    selected_images = images[:max_images]  
    image_tensors = [transform(Image.open(img_path)).unsqueeze(0) for img_path in selected_images]
    stacked_tensors = torch.cat(image_tensors, dim=0)          
    return stacked_tensors 
  

folders = [os.path.join(base_path, folders_class[class_index]) for class_index in range(0,class_number) if os.path.isdir(os.path.join(base_path,folders_class[class_index]))]  


for folder_index in range(0,class_number):     
    index = torch.arange(0, point_number)

    x = get_images_from_folder(folders[folder_index], point_number) 
    k = x.shape[0]
    target = torch.full((k,), folder_index)                 
    x = x.to(device)  
    [preds, feature] = model.predict(x)
    for p in range(0,k):
      cluster_labels.append(torch.argmax(preds, dim=-1).data.cpu()[0][p])    
      feature_extract.append(feature.cpu().numpy()[p][:]) 
    ground_truth_labels += target.data.cpu().tolist()

torch.save({"ground_truth": ground_truth_labels,"cluster_result":cluster_labels, "feature": feature_extract}, output_dir + "/outcomes.npy")
print("done")
