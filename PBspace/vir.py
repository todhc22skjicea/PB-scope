import torch  
import numpy as np  
import umap  
import matplotlib.pyplot as plt  
import csv  

with open('./configs/drug_repeat.csv', mode='r') as file:  
    reader = csv.reader(file)  
    c_folder = list(reader)  

saved_file_path = r'/home/sdx/drug_model/result_T6/outcomes.npy'
out_path  =r"/home/sdx/drug_model/result_T6/"
data = torch.load(saved_file_path)  
ground_truth_labels = data['ground_truth']  
cluster_result = data['cluster_result']  
cluster_feature = data['feature']  
print('ground_truth_labels',len(ground_truth_labels))
#print('cluster_result',len(cluster_result))
#print('cluster_feature',len(cluster_feature))
reducer = umap.UMAP(n_neighbors=20, min_dist=0.01, n_components=2,random_state=42)  
umap_embedding = reducer.fit_transform(cluster_feature)  
print(umap_embedding.shape)
unique_labels = np.unique(cluster_result)  
plt.figure(figsize=(10, 8))  
plt.rcParams['font.family'] = 'Helvetica'  # 
plt.rcParams['font.size'] = 28  # font
#scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=ground_truth_labels, cmap='RdBu', s=50, alpha=0.8)  
scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c='whitesmoke', s=100, alpha=0.7)  
#for i, txt in enumerate(ground_truth_labels):  
#    plt.annotate(txt, (umap_embedding[i, 0], umap_embedding[i, 1]),fontsize=16)  


plt.colorbar(scatter, label='Cluster Label')  
#  
#legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")  

plt.xlabel('UMAP 1')  
plt.ylabel('UMAP 2') 
x_lim = plt.xlim()  
y_lim = plt.ylim()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(out_path+'/test.svg', format='svg')
plt.close()
plt.figure(figsize=(10, 8))  

plt.rcParams['font.family'] = 'Helvetica'  # 
plt.rcParams['font.size'] = 28  # font
#scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=ground_truth_labels, cmap='RdBu', s=50, alpha=0.8)  
scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=cluster_result, cmap='viridis', s=100, alpha=0.7)  
#for i, txt in enumerate(ground_truth_labels):  
#    plt.annotate(txt, (umap_embedding[i, 0], umap_embedding[i, 1]),fontsize=16)  

plt.colorbar(scatter, label='Cluster Label')
plt.title('UMAP result')
plt.xlabel('UMAP 1')  
plt.ylabel('UMAP 2') 
x_lim = plt.xlim()  
y_lim = plt.ylim()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
#legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")  
plt.savefig(out_path+'/test2.svg', format='svg')
plt.close()

for index in range(250,564):
  count1 = 0
  count2 = 0
  count3 = 0
  plt.figure(figsize=(10, 8)) 
  plt.rcParams['font.family'] = 'Helvetica'  # Helvetica
  plt.rcParams['font.size'] = 28  # font
  scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1],label='Group 2',c='whitesmoke', s=100, alpha=0.7)

  for number in range(0,len(cluster_result)):
    if c_folder[0][ground_truth_labels[number]] == "MG132":
      if count1 == 0:
        scatter = plt.scatter(umap_embedding[number, 0], umap_embedding[number, 1],label='MG132',c='indigo',s=100, alpha=0.7)
        count1 = count1+1
      else:
        scatter = plt.scatter(umap_embedding[number, 0], umap_embedding[number, 1],c='indigo',s=100, alpha=0.7)
  for number in range(0,len(cluster_result)):
    if (c_folder[0][ground_truth_labels[number]] == "DMSO" or c_folder[0][ground_truth_labels[number]] == "NA"):
      if count2 == 0:
        scatter = plt.scatter(umap_embedding[number, 0], umap_embedding[number, 1],label='negative control',c='goldenrod',s=100, alpha=0.7)
        count2 = count2+1
      else:
        scatter = plt.scatter(umap_embedding[number, 0], umap_embedding[number, 1],c='goldenrod',s=100, alpha=0.7)
  #for number in range(0,len(cluster_result)):
  #  if ground_truth_labels[number] == 282:
  #    scatter = plt.scatter(umap_embedding[number, 0], umap_embedding[number, 1],c='goldenrod',s=30, alpha=0.5)   
  for number in range(0,len(cluster_result)):

    if ground_truth_labels[number] == index:
      if count3 == 0:
        scatter = plt.scatter(umap_embedding[number, 0], umap_embedding[number, 1],label=c_folder[0][index],c='mediumvioletred',s=100, alpha=0.7)
        count3 = count3+1
      else:
        scatter = plt.scatter(umap_embedding[number, 0], umap_embedding[number, 1],c='mediumvioletred',s=100, alpha=0.7)      
        

  plt.xlabel('UMAP 1')  
  plt.ylabel('UMAP 2') 
  x_lim = plt.xlim()  
  y_lim = plt.ylim()
  ax = plt.gca()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.set_xlabel('')
  ax.set_ylabel('')
  ax.set_xticks([])
  ax.set_yticks([])
  plt.title(c_folder[0][index])
  
  plt.xlabel('UMAP 1')  
  plt.ylabel('UMAP 2')  
  plt.legend()
  plt.xlim(x_lim)  
  plt.ylim(y_lim)  
  
  path = out_path+'/'+c_folder[0][index]+'.png'
  
  plt.savefig(path, format='png')  
  plt.close()

print('done')
