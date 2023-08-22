from typing_extensions import final
import torch
from torch._C import ThroughputBenchmark
import torch.nn.functional as F
import math 

   

'''
loss functions
'''
LOG_EPSILON = 1e-5

def neg_log(x):
    return - torch.log(x + LOG_EPSILON)

def log_loss(preds, targs):
    return targs * neg_log(preds)

def expected_positive_regularizer(preds, expected_num_pos, norm='2'):
    # Assumes predictions in [0,1].
    if norm == '1':
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == '2':
        reg = (preds.sum(1).mean(0) - expected_num_pos)**2
    else:
        raise NotImplementedError
    return reg

def loss_epr(logits, observed_labels, P):
    # unpack:
    preds = torch.sigmoid(logits)
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss w.r.t. observed positives:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # compute regularizer: 
    reg_loss = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    return loss_mtx.mean() + reg_loss

def loss_an(logits, observed_labels):

    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_matrix = F.binary_cross_entropy_with_logits(logits, observed_labels, reduction='none')
    corrected_loss_matrix = F.binary_cross_entropy_with_logits(logits, torch.logical_not(observed_labels).float(), reduction='none')
    return loss_matrix, corrected_loss_matrix

def loss_EM_APL(batch, P):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']

    # input validation:
    assert torch.min(observed_labels) >= -1

    loss_mtx = torch.zeros_like(preds)

    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = -P['alpha'] * (
            preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
        )

    soft_label = -observed_labels[observed_labels < 0]
    loss_mtx[observed_labels < 0] = P['beta'] * (
            soft_label * neg_log(preds[observed_labels < 0]) +
            (1 - soft_label) * neg_log(1 - preds[observed_labels < 0])
        )
    return loss_mtx, None
'''
top-level wrapper
'''

def compute_batch_loss(logits, label_vec, P): 
     
    assert logits.dim() == 2
    
    batch_size = int(logits.size(0))
    num_classes = int(logits.size(1))
    

    if P['dataset'] == 'OPENIMAGES':
        unobserved_mask = (label_vec == -1)
    else:
        unobserved_mask = (label_vec == 0)
    
    # compute loss for each image and class:
    loss_matrix, corrected_loss_matrix = loss_an(logits, label_vec.clip(0))

    correction_idx = [torch.Tensor([]), torch.Tensor([])]

    if P['clean_rate'] == 1 or P['clean_rate'] <= P['delta_rate']: # if epoch is 1, do not modify losses
        final_loss_matrix = loss_matrix
    else:
        if P['largelossmod_scheme'] == 'LL-Cp':
            k = math.ceil(batch_size * num_classes * P['delta_rel'])
        else:
            k = math.ceil(batch_size * num_classes * (1-P['clean_rate']))
    
        unobserved_loss = unobserved_mask.bool() * loss_matrix
        topk = torch.topk(unobserved_loss.flatten(), k)
        topk_lossvalue = topk.values[-1]
        correction_idx = torch.where(unobserved_loss >= topk_lossvalue)


        if P['largelossmod_scheme'] in ['LL-Ct', 'LL-Cp']:
            final_loss_matrix = torch.where(unobserved_loss >= topk_lossvalue, corrected_loss_matrix, loss_matrix)
        else:
            zero_loss_matrix = torch.zeros_like(loss_matrix)
            final_loss_matrix = torch.where(unobserved_loss >= topk_lossvalue, zero_loss_matrix, loss_matrix)
                
    main_loss = P['beta'] * final_loss_matrix.mean() + (1 - P['beta']) * loss_epr(logits, label_vec.clip(0), P)
    
    return main_loss, correction_idx



'''
helper functions
'''


LOG_EPSILON = 1e-5


def neg_log(x):
    return - torch.log(x + LOG_EPSILON)