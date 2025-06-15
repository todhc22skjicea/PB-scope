from utils.logger import Logger
from utils.misc import export_fn
import torch
from torch.cuda.amp import autocast, GradScaler
import os
from engine.criterion import clustering_accuracy_metrics

import numpy as np
import matplotlib.pyplot as plt
import csv
@export_fn
class Trainer:
    def __init__(self, model, optimizer, args, logger:Logger):
        self.train_step=0
        self.epoch=0
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.logger = logger
        self.device = args.gpu
        self.best = 0
        self.scaler = GradScaler()
        self.mixed_precision = args.__dict__.get("mixed_precision", True)
        self.logger.print(f"Mixed precision: {'ON' if self.mixed_precision else 'OFF'}")

    def train_epoch(self, train_dataloader, eval_dataloader, print_interval=100, eval=True,pretrain=True,mixup=False):
        device = self.args.gpu
        #model = torch.load(self.args.output_dir + "/model.pth")
        self.model.train()
        epoch_steps = len(train_dataloader)
        if self.epoch == 0 & pretrain == True:
          model = torch.load(self.args.output_dir + "/model.pth")
        
        for batch_id, batch in enumerate(train_dataloader):
            self.optimizer.zero_grad(set_to_none=True)

            idx, samples, annotations = batch
            
            if self.args.clustering_framework == "cc":

                v1 = samples[:, 0].to(device,non_blocking=True)
                v2 = samples[:, 1].to(device,non_blocking=True)

                if mixup: 
                    mixup_lambda = 0.9
                    shuffled_indices = torch.randperm(v1.size(0), device=device) 
             
                    v1_mixed = mixup_lambda* v1 + (1 - mixup_lambda)* v1[shuffled_indices]
                    v2_mixed = mixup_lambda* v2 + (1 - mixup_lambda)* v2[shuffled_indices]
    
                    with autocast(self.mixed_precision):
                        loss, metrics_dict = self.model(v1_mixed, v2_mixed)  
                else:
                    with autocast(self.mixed_precision):
                        loss, metrics_dict = self.model(v1, v2)
                        

 
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

              
            self.logger.log(metrics_dict)
            if print_interval>0 and batch_id%print_interval==0:
                self.logger.print_epoch_progress(batch_id, epoch_steps, self.epoch, self.args.epochs)
            if self.epoch == (self.args.epochs - 1):
                torch.save(self.model, self.args.output_dir + "/model.pth")

        if eval:
            self.logger.print(f"Evaluating")
            self.model.eval()
            cluster_labels = []
            ground_truth_labels = []
            ground_truth_labels_select = []
            cluster_labels_select = []
            feature_extract = []
            confidence = 0
            samples = 0
            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    if step % 50 == 0:
                        self.logger.print(f"Eval. step {step} of {len(eval_dataloader)}")
                    index, x, target = batch                   
                    x = x.cuda(self.device)
                    [preds, feature] = self.model.predict(x)
                    confidence += preds.max(-1)[0].sum(-1).mean()
                    samples += x.shape[0]

                    cluster_labels.append(torch.argmax(preds, dim=-1).data.cpu())
                    ground_truth_labels.append(target.data.cpu()) 

                    numbers = list(range(target.size(0)))  
                    selected_numbers =np.random.choice(numbers,size = 100)

                    for index_cluster in selected_numbers:
                      ground_truth_labels_select.append(target.data.cpu()[index_cluster])
                      cluster_labels_select.append(torch.argmax(preds, dim=-1).data.cpu()[0][index_cluster])
                      feature_extract.append(feature.cpu().numpy()[index_cluster,:])

            ground_truth_labels = torch.cat(ground_truth_labels, dim=0)
            cluster_labels = torch.cat(cluster_labels, dim=1)
            ground_truth_labels_select = torch.tensor(ground_truth_labels_select)

            metrics_ = clustering_accuracy_metrics(cluster_labels, ground_truth_labels)
            eval_metrics = {"eval_confidence": confidence / samples}
            
            for k, v in metrics_.items():
                eval_metrics[f"{k}_eval"] = v
            self.logger.log(eval_metrics)



        self.logger.epoch_end(self.epoch, self.args.epochs)
        self.epoch+=1
