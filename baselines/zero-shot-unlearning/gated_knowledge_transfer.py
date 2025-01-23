import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *
from models import AllCNN
from metrics import *
from unlearn import *

def initialize_environment(seed=100):
    torch.manual_seed(seed)

def setup_model(device, model_path):
    model = AllCNN(n_channels=1).to(device=device)
    model.load_state_dict(torch.load(model_path))
    return model

def setup_generator_and_optimizer(n_repeat_batch, device):
    generator = LearnableLoader(n_repeat_batch=n_repeat_batch, num_channels=1, device=device).to(device=device)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.001)
    scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_generator, mode='min', factor=0.5, patience=2, verbose=True
    )
    return generator, optimizer_generator, scheduler_generator

def setup_student_and_optimizer(device):
    student = AllCNN(n_channels=1).to(device=device)
    optimizer_student = torch.optim.Adam(student.parameters(), lr=0.001)
    scheduler_student = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_student, mode='min', factor=0.5, patience=2, verbose=True
    )
    return student, optimizer_student, scheduler_student

def save_checkpoint(generator, student, generator_path, student_path, step):
    torch.save(generator.state_dict(), os.path.join(generator_path, f"{step}.pt"))
    torch.save(student.state_dict(), os.path.join(student_path, f"{step}.pt"))

def train_generator(generator, student, model, optimizer_generator, x_pseudo, KL_temperature):
    student_logits, *_ = student(x_pseudo)
    teacher_logits, *_ = model(x_pseudo)
    generator_loss = KT_loss_generator(student_logits, teacher_logits, KL_temperature=KL_temperature)
    optimizer_generator.zero_grad()
    generator_loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)
    optimizer_generator.step()
    return generator_loss.cpu().detach()

def train_student(student, model, optimizer_student, x_pseudo, KL_temperature, AT_beta):
    with torch.no_grad():
        teacher_logits, teacher_activations = model(x_pseudo)
    student_logits, student_activations = student(x_pseudo)
    student_loss = KT_loss_student(student_logits, student_activations, teacher_logits, teacher_activations,
                                   KL_temperature=KL_temperature, AT_beta=AT_beta)
    optimizer_student.zero_grad()
    student_loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
    optimizer_student.step()
    return student_loss.cpu().detach()

def gtk_unlearn(student, model, generator, forget_valid_dl, retain_valid_dl, device, total_n_pseudo_batches=4000,
                n_generator_iter=1, n_student_iter=10, threshold=0.01, KL_temperature=1, AT_beta=250,
                generator_path="path_to_generator_checkpoints", student_path="path_to_student_checkpoints"):
    save_checkpoint(generator, student, generator_path, student_path, step=0)

    idx_pseudo, n_pseudo_batches, zero_count = 0, 0, 0
    running_gen_loss, running_stu_loss = [], []
    n_repeat_batch = n_generator_iter + n_student_iter

    while n_pseudo_batches < total_n_pseudo_batches:
        x_pseudo = generator.__next__()
        preds, *_ = model(x_pseudo)
        mask = (torch.softmax(preds.detach(), dim=1)[:, 0] <= threshold)
        x_pseudo = x_pseudo[mask]

        if x_pseudo.size(0) == 0:
            zero_count += 1
            if zero_count > 100:
                print("Generator stopped producing valid data points. Resetting to checkpoint.")
                last_checkpoint = max(0, (n_pseudo_batches // 50 - 1) * 50)
                generator.load_state_dict(torch.load(os.path.join(generator_path, f"{last_checkpoint}.pt")))
            continue
        zero_count = 0

        if idx_pseudo % n_repeat_batch < n_generator_iter:
            gen_loss = train_generator(generator, student, model, optimizer_generator, x_pseudo, KL_temperature)
            running_gen_loss.append(gen_loss)

        elif idx_pseudo % n_repeat_batch < n_generator_iter + n_student_iter:
            stu_loss = train_student(student, model, optimizer_student, x_pseudo, KL_temperature, AT_beta)
            running_stu_loss.append(stu_loss)

        if (idx_pseudo + 1) % n_repeat_batch == 0:
            if n_pseudo_batches % 50 == 0:
                MeanGLoss, MeanSLoss = np.mean(running_gen_loss), np.mean(running_stu_loss)
                running_gen_loss, running_stu_loss = [], []

                scheduler_student.step(evaluate(student, retain_valid_dl, device=device))

                save_checkpoint(generator, student, generator_path, student_path, step=n_pseudo_batches)

            n_pseudo_batches += 1

        idx_pseudo += 1
    return student

# Example initialization
initialize_environment()
device = 'cuda'
generator_path = "path_to_generator_checkpoints"
student_path = "path_to_student_checkpoints"

model = setup_model(device, "AllCNN_MNIST_ALL_CLASSES.pt")
generator, optimizer_generator, scheduler_generator = setup_generator_and_optimizer(n_repeat_batch=11, device=device)
student, optimizer_student, scheduler_student = setup_student_and_optimizer(device=device)
