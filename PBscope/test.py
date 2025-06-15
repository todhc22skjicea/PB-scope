from utils.logger import Logger
from utils.misc import export_fn
import torch
from torch.cuda.amp import autocast, GradScaler
import os
import torchvision.transforms as transforms 
from engine.criterion import clustering_accuracy_metrics
import numpy as np 
import random  
import umap  
from PIL import Image 
from sklearn.metrics import adjusted_rand_score, pairwise_distances
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "/home/drug_model/result_T6"
base_path = "/home/drugdataset_final/dataset_t6/test"
print(f"Evaluating")
model = torch.load("/home/drug_model/model/T6/CC/model.pth")
model.eval()
#folders_class = ["T1648", "T1297", "T0437", "T0374", "T0429", "T0373", "T0461", "T0433", "T0364", "T0038","T0030", "T0129", "T0147", "T0163", "T0256", "T0304", "T0314", "T0327", "T0342", "T0492","T0498", "T0520", "T0263", "T0374L", "T2215", "T0973", "T1020", "T1056", "T1090", "T1144","T1146", "T1158", "T1159", "T0239", "T1291", "T2205", "T1181", "T1188", "T1210", "T1222","T0167", "T2597", "T0610", "T0646", "T2565", "T0711", "T0704", "T0692", "T0679", "T0773","T0678", "T0449", "T2508", "T0445", "T1636", "T2586","MG132","TG","DMSO","NA"]#Group1
#folders_class = ["T0772", "T0740", "T0875", "T0891", "T2995", "T2827", "T2858", "T0800", "T0801", "T0809", "T0858", "T0860", "T2546", "T0392", "T0928", "T0933", "T1410", "T1418", "T1431", "T1439", "T1448", "T1452", "T1454", "T1477", "T1524", "T1537", "T1546", "T1558", "T1563", "T1661", "T1737", "T2175", "T2144", "T2399", "T1630", "T1621", "T1639", "T1642", "T1659", "T1660", "T1684", "T2381", "T2364", "T2382", "T2372", "T2369", "T2145", "T2115", "T2587", "T2532", "T1835", "T0335", "T8222", "T1085L", "T0152", "T2303", "MG132", "TG", "DMSO","NA"] #Group2
#folders_class = ["T2534", "T0097L", "T2148", "T2490", "T2483", "T3060", "T3059", "T1995", "T2325", "T2328", "T2920", "T2984", "T2851", "T6218", "T6230", "T3091", "T6227", "T2946", "T1656", "T1266", "T0080", "T0078", "T1038", "T1670", "T1829", "T1743", "T1777", "T1797", "T1912", "T1784", "T1785", "T1791", "T1792", "T1506", "T2220", "T0093L", "T2677", "T2456", "T2509", "T2539", "T2500", "T2066", "T2125", "T2485", "T2397", "T3067", "T2609", "T2656", "T1963", "T1921", "T6199", "T8387", "T14998", "T6460", "T1260", "T8151", "MG132", "TG", "DMSO","NA"] #Group3
#folders_class = ["T1929", "T1936", "T1975", "T1894", "T1903", "T1988", "T3061", "T3211", "T6321", "T3269", "T3616", "T3402", "T3380", "T3626", "T3625", "T6115", "T3623", "T3634", "T3678", "T6165", "T6019", "T3726", "T6867", "T6121", "T6120", "T6280", "T6302", "T6758", "T6345", "T2S0007", "T6169", "T2P2923", "T6723", "T6588", "T4168", "T6156", "T6020", "T4332", "T7503", "T6880", "T6101", "T4575", "T4409", "T6487", "T6674", "T4749", "T4976", "T5001", "T5030", "T5109", "T3603", "T1633", "T0745", "T2796", "T7175", "T3O2749", "MG132", "TG", "DMSO","NA"] #Group4
#folders_class = ["T5171", "T5177", "T5462", "T5882", "T7094", "T6930", "T5995", "T7584", "T7394", "T7486", "T7861", "T8132", "T5857", "T7604", "T8402", "T6914", "T8399", "T6475", "TQ0277", "T1791L", "T8482", "T12401", "TQ0210", "T8474", "TQ0319", "T8541", "T15732", "T8651", "T8654", "T2147", "T12317", "T8684", "T8825", "T22235", "TQ0064", "T0979", "T10358", "T19965", "T12311", "T1756L", "T22324", "T10585", "T20029", "T12594", "T15675", "T0247", "T1035", "T0878", "T2854", "T5016", "T4883", "T0194", "T2211", "T0033", "T13202","NA", "MG132", "TG", "DMSO","NA"] #Group5

folders_class = ["T1648", "T1297", "T0437", "T0374", "T0429", "T0373", "T0461", "T0433", "T0364", "T0038","T0030", "T0129", "T0147", "T0163", "T0256", "T0304", "T0314", "T0327", "T0342", "T0492","T0498", "T0520", "T0263", "T0374L", "T2215", "T0973", "T1056", "T1090", "T1144","T1146", "T1158", "T1159", "T0239", "T1291", "T2205", "T1181", "T1188", "T1210", "T1222","T0167", "T2597", "T0610", "T0646", "T2565", "T0711", "T0704", "T0692", "T0679", "T0773","T0678", "T0449", "T2508", "T0445", "T1636", "T2586","T0772", "T0740", "T0875", "T0891", "T2995", "T2827", "T2858", "T0800", "T0801", "T0809", "T0858", "T0860", "T2546", "T0392", "T0928", "T0933", "T1410", "T1418", "T1431", "T1439", "T1448", "T1452", "T1454", "T1477", "T1524", "T1537", "T1546", "T1558", "T1563", "T1661", "T1737", "T2175", "T2144", "T2399", "T1630", "T1621", "T1639", "T1642", "T1659", "T1660", "T1684", "T2381", "T2364", "T2382", "T2372", "T2369", "T2145", "T2115", "T2587", "T2532", "T1835", "T0335", "T8222", "T1085L", "T0152", "T2303", "T2534", "T0097L", "T2148", "T2490", "T2483", "T3060", "T3059", "T1995", "T2325", "T2328", "T2920", "T2984", "T2851", "T6218", "T6230", "T3091", "T6227", "T2946", "T1656", "T1266", "T0080", "T0078", "T1038", "T1670", "T1829", "T1743", "T1777", "T1797", "T1912", "T1784", "T1785", "T1791", "T1792", "T1506", "T2220", "T0093L", "T2677", "T2456", "T2509", "T2539", "T2500", "T2066", "T2125", "T2485", "T2397", "T3067", "T2609", "T2656", "T1963", "T1921", "T6199", "T8387", "T14998", "T6460", "T1260", "T8151", "T1929", "T1936", "T1975", "T1894", "T1903", "T1988", "T3061", "T3211", "T6321", "T3269", "T3616", "T3402", "T3380", "T3626", "T3625", "T6115", "T3623", "T3634", "T3678", "T6165", "T6019", "T3726", "T6867", "T6121", "T6120", "T6280", "T6302", "T6758", "T6345", "T2S0007", "T6169", "T2P2923", "T6723", "T6588", "T4168", "T6156", "T6020", "T4332", "T7503", "T6880", "T6101", "T4575", "T4409", "T6487", "T6674", "T4749", "T4976", "T5001", "T5030", "T5109", "T3603", "T1633", "T0745", "T2796", "T7175", "T3O2749", "T5171", "T5177", "T5462", "T5882", "T7094", "T6930", "T5995", "T7584", "T7394", "T7486", "T7861", "T8132", "T5857", "T7604", "T8402", "T6914", "T8399", "T6475", "TQ0277", "T1791L", "T8482", "T12401", "TQ0210", "T8474", "TQ0319", "T8541", "T15732", "T8651", "T8654", "T2147", "T12317", "T8684", "T8825", "T22235", "TQ0064", "T0979", "T10358", "T19965", "T12311", "T1756L", "T22324", "T10585", "T20029", "T12594", "T15675", "T0247", "T1035", "T0878", "T2854", "T5016", "T4883", "T0194", "T2211", "T0033", "T13202","MG132", "TG", "DMSO","NA"] #Group_all
'''


folders_class = ["T2369","exp2_T2946","T0304","exp2_T2534","T3603","T7503","exp2_T10585","exp2_T2328","exp2_T1524","T1784","T3067","exp2_T0147","exp2_T15732","exp2_T2597","T0773","T0461","exp2_T0520","exp2_T0374","T8402","exp2_T2508","exp2_T1085L","T2399","T0342","exp2_T0247","exp2_T2372","T1633","T5882","exp2_T2148","T1563","T1158","T5177","T0263","exp2_T1912","exp2_T3059","exp2_TQ0319","exp2_T4168","exp2_T1260","T4409","exp2_T2858","exp2_T1188","exp2_T2485","exp2_T4409","T6165","exp2_T0152","exp2_T0239","T1085L","exp2_T0801","exp2_T7584","T6588","exp2_T2325","exp2_T2303","T3678","exp2_T8684","T2325","T5030","exp2_T7861","exp2_T5030","exp2_T1056","T3059","T0364","T8825","exp2_T1431","exp2_T2509","T0093L","T3726","exp2_T0449","T1975","T8482","exp2_T5995","exp2_T6302","T8387","T1743","exp2_T2125","exp2_T1181","exp2_T0878","T2546","exp2_T1684","T1756L","T12317","exp2_T0364","T2456","exp2_T2456","T1656","T10585","exp2_T2382","T6227","T1639","T6156","exp2_T1975","T0080","T2586","T7394","exp2_T2995","exp2_T8151","exp2_T3380","T1454","exp2_NA","exp2_T7486","exp2_T0093L","exp2_T3678","exp2_T1648","exp2_T0646","T12401","exp2_T2364","T6230","exp2_T12317","exp2_T10358","exp2_T3O2749","exp2_T0163","exp2_T0097L","exp2_T8482","exp2_T7094","exp2_T4332","T6880","exp2_T2205","T2215","T2364","exp2_T1784","T0152","T3616","DMSO","T5995","exp2_T6156","T1929","T1090","T8474","exp2_T2539","exp2_T6019","T0928","T0078","T5109","T0646","T0194","T2485","T6020","T1791","exp2_T7604","exp2_T1144","T0038","exp2_T6169","T2656","exp2_T0256","T0327","exp2_T0928","exp2_T0342","exp2_T6674","T0392","T1056","T1921","T1661","TQ0319","T1537","exp2_T1639","exp2_T1159","T2483","exp2_T2S0007","exp2_T1791","exp2_T2483","T0097L","exp2_T0335","exp2_T1537","T1222","T1648","exp2_T0745","T6120","T3626","T0809","T0800","T0129","exp2_T2587","exp2_T6230","T2532","exp2_T0429","exp2_T3603","T1291","T6930","exp2_T1477","exp2_T1670","T0256","T7175","exp2_T2500","T0740","T2597","T2125","exp2_T2532","T0030","TQ0277","T3269","T2P2923","T6121","T1146","exp2_T5109","exp2_T0772","exp2_T0194","exp2_T0979","exp2_T6723","T1418","T1452","T4883","exp2_T1791L","exp2_T1936","T1297","exp2_T0740","T0239","T1038","T3634","exp2_T0498","exp2_T2851","T0860","T2381","exp2_TQ0064","T2984","exp2_T1038","exp2_T2381","exp2_T0711","exp2_T0033","T0033","exp2_T2220","T8132","exp2_T2565","exp2_T0078","exp2_T1563","exp2_T2211","T2539","exp2_T6121","exp2_T7503","exp2_T6588","exp2_T7175","exp2_T8651","T0314","T2148","T2144","T1431","T1995","exp2_T2215","exp2_T3623","T7094","exp2_T3091","exp2_T22324","T2920","exp2_T0433","T2796","T4332","exp2_T1630","exp2_T2P2923","exp2_T1454","T1558","exp2_T1452","exp2_T1210","exp2_T3402","exp2_T2175","exp2_TG","T0772","exp2_T1921","T5016","exp2_T6321","T6758","exp2_T6460","exp2_T6020","exp2_T5001","exp2_T0704","exp2_T6867","exp2_T6227","exp2_T1656","T7861","exp2_T2144","exp2_T3626","exp2_T0679","exp2_T0875","exp2_T6475","T6019","T2220","exp2_T0030","exp2_T0692","T0374","T6674","T7584","T0801","exp2_T1829","T2147","T6475","exp2_T8474","exp2_T1988","T2205","T6199","T22324","T1621","T12311","T1477","exp2_T1660","exp2_T0304","exp2_T1929","T0679","T8541","exp2_T0773","T0979","T2854","exp2_T4749","T0498","exp2_T1035","T1144","exp2_T8825","exp2_T6165","exp2_T4575","exp2_T19965","exp2_T0080","T2500","exp2_T5177","exp2_DMSO","T2851","T3091","T1936","exp2_T1661","exp2_T8132","T2S0007","exp2_T1903","exp2_T1636","exp2_T1995","exp2_T2147","exp2_T6218","T2609","T1791L","exp2_T3061","T7604","exp2_T2369","T2587","exp2_T0327","exp2_T0038","exp2_T1642","T2372","exp2_T1090","T2328","exp2_T13202","T2509","T1410","exp2_T1448","exp2_T1297","exp2_T2920","T1260","T1894","exp2_T1743","exp2_T1777","T2303","T0520","T0429","exp2_T6880","T12594","T8651","T1963","exp2_TQ0210","T2565","exp2_T0437","exp2_T0461","exp2_T7394","exp2_T2066","exp2_T0610","T0437","T22235","T6218","T8684","T6302","T3380","exp2_T6199","T1670","T3402","T0891","exp2_T12594","T0374L","T10358","T5171","T1684","T1829","exp2_T2115","exp2_T0891","exp2_T3067","exp2_T3060","exp2_T5882","T0433","T5857","exp2_T0263","T3211","exp2_T0374L","exp2_T1792","exp2_T0858","T20029","exp2_T2145","T1903","T6345","T4168","T0373","exp2_T6280","T0858","exp2_TQ0277","exp2_T6758","exp2_T4976","exp2_T6115","exp2_T1410","T0147","T3060","exp2_T3269","T8222","exp2_T8402","exp2_T1785","T0878","NA","T0678","T1642","TG","T1266","T1660","T1210","exp2_T1546","T1439","T1737","exp2_T12401","T8151","T1188","exp2_T0314","T2946","T5001","exp2_T3616","exp2_T6930","exp2_T8654","T0704","exp2_T5171","T1524","T0335","T4749","exp2_T2586","exp2_T2984","T3061","exp2_T1222","exp2_T2609","exp2_T3625","exp2_T0167","exp2_T22235","T13202","exp2_T8222","T1448","exp2_T2490","exp2_T0809","exp2_T5857","T19965","exp2_T5462","exp2_T1506","exp2_T0373","T2066","T0933","T6460","exp2_MG132","T0692","T1159","T1506","T2175","T3O2749","exp2_T6914","T2677","T3625","exp2_T1621","T6280","exp2_T1963","exp2_T6345","exp2_T1835","exp2_T8541","exp2_T0392","T2858","T2397","exp2_T8399","exp2_T0678","T2534","exp2_T5016","T1785","T1792","exp2_T0860","T4976","T15675","exp2_T3211","exp2_T6120","exp2_T1146","T8654","exp2_T6101","exp2_T1797","T0745","exp2_T2854","T8399","exp2_T15675","TQ0210","T2508","exp2_T2656","T0610","T1988","exp2_T0445","T0445","exp2_T1418","exp2_T2399","exp2_T1894","T0167","T6914","exp2_T3634","exp2_T2546","exp2_T1158","T0163","exp2_T1756L","TQ0064","T3623","exp2_T1558","exp2_T20029","T1797","T1912","T1181","exp2_T1266","T0973","exp2_T6487","T0711","T6115","T6723","exp2_T1633","MG132","exp2_T8387","T6101","T2995","exp2_T0973","T7486","exp2_T3726","T1659","exp2_T2827","T0449","T1546","exp2_T0129","T15732","T2827","T6169","exp2_T12311","T0492","T6867","exp2_T1737","exp2_T0933","exp2_T1291","exp2_T2677","exp2_T1659","T2490","exp2_T0492","T6321","T1636","T1035","exp2_T4883","T1835","T2115","T1777","exp2_T2796","T0247","T2382","T6487","T2211","T5462","T1630","exp2_T0800","exp2_T2397","exp2_T1439","T4575","T2145","T14998","exp2_T14998","T0875"]
'''
class_number = len(folders_class)
#print(class_number)
point_number = 30
cluster_labels = []
ground_truth_labels = []
ground_truth_labels_select = []
cluster_labels_select = []
feature_extract = [] 
scope_all_by_label = {}
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
    #target = torch.full((class_number,), folder_index) 
    #print(folder_index)
    #print(len(folders))
    x = get_images_from_folder(folders[folder_index], point_number) 
    k = x.shape[0]
    target = torch.full((k,), folder_index)                 
    x = x.to(device)  
    [preds, feature] = model.predict(x)
    scope = AggregationModel(feature)
    #print(preds.shape)
    #print(torch.argmax(preds, dim=-1).data.cpu())
    for p in range(0,k):
      cluster_labels.append(torch.argmax(preds, dim=-1).data.cpu()[0][p])    
      feature_extract.append(feature.cpu().numpy()[p][:]) 
      #print(scope.shape)
      scope_value = scope.detach().cpu().numpy()[0] 
      label = target.data.cpu().tolist()[p]      
      if label not in scope_all_by_label:
        scope_all_by_label[label] = []
      scope_all_by_label[label].append(scope_value)
    ground_truth_labels += target.data.cpu().tolist()      


average_scope_by_label = {}
for label, scopes in scope_all_by_label.items():
    average_scope_by_label[label] = np.mean(scopes)
 

for label, avg_scope in average_scope_by_label.items():
    drug_name = folders_class[label]  
    print(f"drugname: {drug_name}, Scope: {avg_scope}")

 

average_scope_by_label = {label: np.mean(scopes) for label, scopes in scope_all_by_label.items()}
torch.save({"ground_truth": ground_truth_labels,"cluster_result":cluster_labels, "feature": feature_extract}, output_dir + "/outcomes.npy")
