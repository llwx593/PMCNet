import numpy as np

def get_lr(global_step,
           total_epochs,
           steps_per_epoch,
           lr_init=0.001):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs

    for i in range(int(total_steps)):    
        lr = lr_init * (1.0 - global_step / total_steps) ** 0.9
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]
        
    return learning_rate