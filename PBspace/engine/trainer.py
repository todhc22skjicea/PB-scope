from utils.logger import Logger
from utils.misc import export_fn
import torch
from torch.cuda.amp import autocast, GradScaler
import os
from engine.criterion import clustering_accuracy_metrics
import numpy as np
import matplotlib.pyplot as plt
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

    def train_epoch(self, train_dataloader, eval_dataloader, print_interval=100, eval=True,pretrain=False):
        device = self.args.gpu
        self.model.train()
        loss_history = {
            "loss_cc": [],
            "loss_ce": [], 
            "loss_ne": [],
            "loss": []
        }

        epoch_steps = len(train_dataloader)
        if self.epoch == 0 & pretrain == True:
          model = torch.load(self.args.output_dir + "/model.pth")
        
        for batch_id, batch in enumerate(train_dataloader):
            self.optimizer.zero_grad(set_to_none=True)

            idx, samples, annotations = batch
            if self.args.clustering_framework == "cc":
                v1 = samples[:, 0].to(device,non_blocking=True)
                v2 = samples[:, 1].to(device,non_blocking=True)
                with autocast(self.mixed_precision):
                    loss, metrics_dict = self.model(v1, v2)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            metrics_dict.update({"loss":loss})


              
            self.logger.log(metrics_dict)
            if print_interval>0 and batch_id%print_interval==0:
                self.logger.print_epoch_progress(batch_id, epoch_steps, self.epoch, self.args.epochs)
        for key in metrics_dict:
            loss_history[key].append(metrics_dict[key].item())
        if self.epoch == (self.args.epochs - 1):
            training_epochs = range(self.args.epochs-1)
                          
            plt.figure(figsize=(10, 6))
            plt.plot(training_epochs, loss_history['loss_cc'], label='Cluster Loss')
            plt.plot(training_epochs, [a + b for a, b in zip(loss_history['loss_ce'],loss_history['loss_ne'])], label='Instance Loss')
            plt.plot(training_epochs, loss_history['loss'], label='Total Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.args.output_dir+'/Loss.png', format='png')
            plt.close()
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

            #metrics_ = clustering_accuracy_metrics(cluster_labels, ground_truth_labels)
            eval_metrics = {"eval_confidence": confidence / samples}
            '''
            for k, v in metrics_.items():
                eval_metrics[f"{k}_eval"] = v
            self.logger.log(eval_metrics)
            
            if metrics_["max_cluster_acc"] > self.best:
                self.best = metrics_["max_cluster_acc"]
                torch.save(self.model, self.args.output_dir + "/model.pth")
                print("model update")
            '''
            if self.epoch == (self.args.epochs - 1):
                torch.save(self.model, self.args.output_dir + "/model.pth")
                torch.save({"ground_truth": ground_truth_labels_select.cpu().numpy(),"cluster_result":cluster_labels_select, "feature": feature_extract}, self.args.output_dir + "/outcomes.npy")

        self.logger.epoch_end(self.epoch, self.args.epochs)
        self.epoch+=1
