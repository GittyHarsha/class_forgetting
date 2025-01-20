# Necessary Imports
import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from utils import *
from models import AllCNN
from metrics import *
from unlearn import *

torch.manual_seed(100)


#os.makedirs(generator_path)
#os.makedirs(student_path)
device = 'cuda'
idx_pseudo = 0
total_n_pseudo_batches = 4000
n_pseudo_batches = 0
running_gen_loss = []
running_stu_loss = []
threshold = 0.01
KL_temperature = 1
AT_beta = 250
n_generator_iter = 1
n_student_iter = 10
n_repeat_batch = n_generator_iter + n_student_iter
model = AllCNN(n_channels = 1).to(device = device)
model.load_state_dict(torch.load("AllCNN_MNIST_ALL_CLASSES.pt"))

student = AllCNN(n_channels = 1).to(device = device)
generator = LearnableLoader(n_repeat_batch=n_repeat_batch, num_channels = 1, device = device).to(device=device)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.001) 
scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_generator, 
                                                               mode='min', factor=0.5, patience=2, verbose=True)
optimizer_student = torch.optim.Adam(student.parameters(), lr=0.001)
scheduler_student = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_student, \
                                    mode='min', factor=0.5, patience=2, verbose=True)


def gtk_unlearn(student, model, generator, forget_valid_dl, retain_valid_dl, device):
    # saving the generator
    torch.save(generator.state_dict(), os.path.join(generator_path, str(0) + ".pt"))

    # saving the student
    torch.save(student.state_dict(), os.path.join(student_path, str(0) + ".pt"))




    while n_pseudo_batches < total_n_pseudo_batches:
        x_pseudo = generator.__next__()
        preds, *_ = model(x_pseudo)
        mask = (torch.softmax(preds.detach(), dim=1)[:, 0] <= threshold)
        x_pseudo = x_pseudo[mask]
        if x_pseudo.size(0) == 0:
            zero_count += 1
            if zero_count > 100:
                print("Generator Stopped Producing datapoints corresponding to retain classes.")
                print("Resetting the generator to previous checkpoint")
                generator.load_state_dict(torch.load(os.path.join(generator_path, str(((n_pseudo_batches//50)-1)*50) + ".pt")))
            continue
        else:
            zero_count = 0
        
        ## Take n_generator_iter steps on generator
        if idx_pseudo % n_repeat_batch < n_generator_iter:
            student_logits, *student_activations = student(x_pseudo)
            teacher_logits, *teacher_activations = model(x_pseudo)
            generator_total_loss = KT_loss_generator(student_logits, teacher_logits, KL_temperature=KL_temperature)

            optimizer_generator.zero_grad()
            generator_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)
            optimizer_generator.step()
            running_gen_loss.append(generator_total_loss.cpu().detach())


        elif idx_pseudo % n_repeat_batch < (n_generator_iter + n_student_iter):
            
            
            with torch.no_grad():
                teacher_logits, *teacher_activations = model(x_pseudo)

            student_logits, *student_activations = student(x_pseudo)
            student_total_loss = KT_loss_student(student_logits, student_activations, 
                                                teacher_logits, teacher_activations, 
                                                KL_temperature=KL_temperature, AT_beta = AT_beta)

            optimizer_student.zero_grad()
            student_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            optimizer_student.step()
            running_stu_loss.append(student_total_loss.cpu().detach())
            
        if (idx_pseudo + 1) % n_repeat_batch == 0:       
            if((n_pseudo_batches)% 50 == 0):
                MeanGLoss = np.mean(running_gen_loss)
                running_gen_loss = []
                MeanSLoss = np.mean(running_stu_loss)
                running_stu_loss = []

                scheduler_student.step(evaluate(student, retain_valid_dl, device = device))
                
                # saving the generator
                torch.save(generator.state_dict(), os.path.join(generator_path, str(n_pseudo_batches) + ".pt"))
                
                # saving the student
                torch.save(student.state_dict(), os.path.join(student_path, str(n_pseudo_batches) + ".pt"))
                
                
            n_pseudo_batches += 1
            
        idx_pseudo += 1