import torch  
import numpy as np  
import umap  
import matplotlib.pyplot as plt  
import csv  
from matplotlib.colors import LinearSegmentedColormap
with open('./configs/drug_repeat.csv', mode='r') as file:  
    reader = csv.reader(file)  
    c_folder = list(reader)  

Group1 = [ "T0374", "T0374L","T1912","T1995", "T6020","T7394"
]
common_elements = set(c_folder[0]).intersection(Group1)
indices1 = [i for i, element in enumerate(c_folder[0]) if element in common_elements]
Group2 = [
    "T0860","T1448","T1656","T2508","T2509","T3380"
]
common_elements = set(c_folder[0]).intersection(Group2)
indices2 = [i for i, element in enumerate(c_folder[0]) if element in common_elements]
Group3 = [
    "T0801","T1829","T2485","T2677","T6156","T7503"
]
common_elements = set(c_folder[0]).intersection(Group3)
indices3 = [i for i, element in enumerate(c_folder[0]) if element in common_elements]
Group4 = [
    "T1929","T1936","T2S0007","T3211", "T3269", "T3603", "T3616", "T4883","T5171","T5177","T6321",
    "T7486","T7861","T8482"
]
common_elements = set(c_folder[0]).intersection(Group4)
indices4 = [i for i, element in enumerate(c_folder[0]) if element in common_elements]
Group5 = [
     "T12594", "T1791", "T1791L", "T2372", "T6165", "T6588","T6758",
    "T7604","T8387"
]

common_elements = set(c_folder[0]).intersection(Group5)
indices5 = [i for i, element in enumerate(c_folder[0]) if element in common_elements]
Group6 = [
    "DMSO"
]
Group1_exp2 = ["exp2_" + element for element in Group1]
Group2_exp2 = ["exp2_" + element for element in Group2]
Group3_exp2 = ["exp2_" + element for element in Group3]
Group4_exp2 = ["exp2_" + element for element in Group4]
Group5_exp2 = ["exp2_" + element for element in Group5]
Group6_exp2 = ["exp2_" + element for element in Group6]
colors_r = [
    (0.4039, 0.0000, 0.0510),
    (0.5031, 0.0240, 0.0638),
    (0.6022, 0.0480, 0.0766),
    (0.6850, 0.0678, 0.0903),
    (0.7458, 0.0822, 0.1031),
    (0.8105, 0.1081, 0.1197),
    (0.8681, 0.1641, 0.1437),
    (0.9301, 0.2244, 0.1696),
    (0.9540, 0.2971, 0.2145),
    (0.9747, 0.3781, 0.2662)
]
 
colors_p = [
    (0.2471, 0.0000, 0.4902),
    (0.2806, 0.0624, 0.5190),
    (0.3142, 0.1248, 0.5478),
    (0.3514, 0.1949, 0.5808),
    (0.3866, 0.2621, 0.6128),
    (0.4245, 0.3352, 0.6484),
    (0.4597, 0.4056, 0.6852),
    (0.4976, 0.4814, 0.7248),
    (0.5439, 0.5308, 0.7490),
    (0.5956, 0.5807, 0.7731)
]
 
colors_o = [
    (0.4980, 0.1529, 0.0157),
    (0.5604, 0.1769, 0.0141),
    (0.6228, 0.2009, 0.0125),
    (0.7020, 0.2298, 0.0098),
    (0.7835, 0.2585, 0.0066),
    (0.8606, 0.2955, 0.0111),
    (0.8990, 0.3483, 0.0399),
    (0.9403, 0.4052, 0.0709),
    (0.9619, 0.4621, 0.1319),
    (0.9826, 0.5242, 0.2025),
    (0.9922, 0.5793, 0.2729),
    (0.9922, 0.6362, 0.3538),
    (0.9922, 0.6892, 0.4306),
    (0.9922, 0.7477, 0.5253),
    (0.9922, 0.8021, 0.6133),
    (0.9935, 0.8448, 0.6935),
    (0.9951, 0.8800, 0.7639),
    (0.9968, 0.9128, 0.8288),
    (0.9984, 0.9368, 0.8752),
    (1.0000, 0.9608, 0.9216),
    
    
    
]
 
colors_g = [
    (0.0000, 0.2667, 0.1059),
    (0.0000, 0.3322, 0.1331),
    (0.0000, 0.3978, 0.1603),
    (0.0350, 0.4574, 0.1975),
    (0.0910, 0.5054, 0.2375),
    (0.1493, 0.5579, 0.2802),
    (0.1972, 0.6091, 0.3186),
    (0.2489, 0.6642, 0.3599),
    (0.3263, 0.7056, 0.3997),
    (0.4141, 0.7486, 0.4428),
    (0.4909, 0.7854, 0.4923),
    (0.5684, 0.8216, 0.5561),
    (0.9686, 0.9882, 0.9608),
    (0.6390, 0.8542, 0.6152),
    (0.7044, 0.8817, 0.6790),
    (0.7652, 0.9073, 0.7381),
    (0.8201, 0.9296, 0.7953),
    (0.8681, 0.9488, 0.8464),
    (0.9110, 0.9658, 0.8936),
    (0.9398, 0.9770, 0.9272)



]
 
colors_b = [
    (0.0314, 0.1882, 0.4196),
    (0.0314, 0.2410, 0.4980),
    (0.0314, 0.2938, 0.5763),
    (0.0564, 0.3496, 0.6368),
    (0.0963, 0.4008, 0.6767),
    (0.1426, 0.4563, 0.7166),
    (0.1954, 0.5091, 0.7438),
    (0.2522, 0.5660, 0.7731),
    (0.3162, 0.6117, 0.7989),
    (0.3868, 0.6600, 0.8264)
]

common_elements = set(c_folder[0]).intersection(Group6)
indices6 = [i for i, element in enumerate(c_folder[0]) if element in common_elements]
saved_file_path = r'/home/drug_model/result_T6/outcomes.npy'
out_path  =r"/home/drug_model/result_T6/"
data = torch.load(saved_file_path)  
ground_truth_labels = data['ground_truth']  
cluster_result = data['cluster_result']  
cluster_feature = data['feature']
#print('ground_truth_labels',len(ground_truth_labels))
#print('cluster_result',len(cluster_result))
#print('cluster_feature',len(cluster_feature))

reducer = umap.UMAP(n_neighbors=20, min_dist=0.01, n_components=2,random_state=42)  
umap_embedding = reducer.fit_transform(cluster_feature)  
#print(umap_embedding.shape)

umap = [umap_embedding[i,:] for i,element in enumerate(ground_truth_labels) if ground_truth_labels[i] in indices1]
unique_labels = np.unique(cluster_result) 
plt.figure(figsize=(10, 10)) 
plt.rcParams['font.family'] = 'Helvetica'  # Helvetica
plt.rcParams['font.size'] = 28  # font
scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c='whitesmoke', s=100, alpha=0.7)  
count1 = np.zeros([20,1])
count2 = np.zeros([20,1])
count3 = np.zeros([20,1])
count4 = np.zeros([20,1])
count5 = np.zeros([20,1])
count6 = np.zeros([20,1])

for number in range(0,len(cluster_result)):
  if c_folder[0][ground_truth_labels[number]] in Group1:
    color_index = Group1.index(c_folder[0][ground_truth_labels[number]])
    if count6[color_index] == 0:
      scatter = plt.scatter(umap_embedding[number, 0], umap_embedding[number, 1],label="exp1",c = "MediumVioletRed",s=100, alpha=0.7)
      count6[color_index] += 1
    else:
      scatter = plt.scatter(umap_embedding[number, 0], umap_embedding[number, 1],c = "MediumVioletRed",s=100, alpha=0.7)
  if c_folder[0][ground_truth_labels[number]] in Group1_exp2:
    color_index = Group1_exp2.index(c_folder[0][ground_truth_labels[number]])
    if count3[color_index] == 0:
      scatter = plt.scatter(umap_embedding[number, 0], umap_embedding[number, 1],label="exp2",c = "Navy",s=100, alpha=0.7)
      count6[color_index] += 1
    else:
      scatter = plt.scatter(umap_embedding[number, 0], umap_embedding[number, 1],c = "Navy",s=100, alpha=0.7)
plt.xlabel('UMAP 1')  
plt.ylabel('UMAP 2') 
x_lim = plt.xlim()  
y_lim = plt.ylim()
ax = plt.gca()

handles, labels = ax.get_legend_handles_labels()
hl_sorted = sorted(zip(handles, labels), key=lambda x: x[1])
handles_sorted, labels_sorted = zip(*hl_sorted)
#ax.legend(handles_sorted, labels_sorted)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('./Group1_repeat.png', bbox_inches = 'tight',pad_inches = 0.05,format='png')
plt.close()

