## EVA Assignment 11
----------------------
## Name : Abhijit Mali
----------------------
## Notes 
---------------------------------------------------------------------------------------------------------------------------
1. Wrote own function for One Cycle Policy -> src->train->trainhelper.py. Traingle plot was done with Learning Rate (y-axis) vs Epochs on X-Axis using matplot lib.

![triangleplot](https://github.com/csharpshooter/EVA/blob/master/A11/images/TrainglePlot.png)

2. Ran model for 25 epochs for lr range test lr range tested from 0.001 to 0.096. Found Max Lr between 0.07 to 0.096.
Min LR = Max LR / 15

![lrrangefinder](https://github.com/csharpshooter/EVA/blob/master/A11/images/lrrangetestgraph.png)

3. Max train accuracy =  92.02, max test accuracy = 90.45.
4. Used following pytorch transforms for augmentation:
  *  RandomRotate(20),
  *  RandomHorizontalFlip,
  *  RandomCrop(size=(32, 32), padding=4),
  *  RandomErasing(scale=(0.08, 0.08), ratio=(1, 1)),
 5. Showing weights at layer 20 and layer 10 along with gradcam outputs along with saliency map.
  
 ![correctsalmap](https://github.com/csharpshooter/EVA/blob/master/A11/images/gradcam/correct_1.png)
 ![misclasssalmap](https://github.com/csharpshooter/EVA/blob/master/A11/images/gradcam/misclassifed_1.png)
 
 7. Added Tensorboard visualization

 ![ModelGraph](https://github.com/csharpshooter/EVA/blob/master/A11/images/ModelGraphTensorBoard.png)
    
---------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------
## Project Structure
--------------------

---------------------------------------------------------------------------------------------------------------------------
## Test and Train, Loss and Accuracy Graphs

![Graphs](https://github.com/csharpshooter/EVA/blob/master/A11/images/traintestgraphs.png)

---------------------------------------------------------------------------------------------------------------------------
## Model Summary

![ModelSummary](https://github.com/csharpshooter/EVA/blob/master/A11/images/modelsummary.png)

---------------------------------------------------------------------------------------------------------------------------
## Logs
0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 0
Learning rate = 0.0068  for epoch:  0
/home/abhijit/.virtualenvs/dl4cv/lib/python3.6/site-packages/torch/nn/_reduction.py:43: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))
Loss=1.8318748474121094 Batch_id=97 Accuracy=22.69: 100%|██████████| 98/98 [00:19<00:00,  4.95it/s]

Test set: Average loss: 0.0037, Accuracy: 3151/10000 (31.51%)

Validation accuracy increased (0.000000 --> 31.510000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 1
Learning rate = 0.0221  for epoch:  1
Loss=1.475935935974121 Batch_id=97 Accuracy=35.73: 100%|██████████| 98/98 [00:22<00:00,  4.36it/s] 

Test set: Average loss: 0.0030, Accuracy: 4954/10000 (49.54%)

Validation accuracy increased (31.510000 --> 49.540000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 2
Learning rate = 0.0374  for epoch:  2
Loss=1.3536574840545654 Batch_id=97 Accuracy=43.35: 100%|██████████| 98/98 [00:33<00:00,  2.90it/s]

Test set: Average loss: 0.0027, Accuracy: 5211/10000 (52.11%)

Validation accuracy increased (49.540000 --> 52.110000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 3
Learning rate = 0.052700000000000004  for epoch:  3
Loss=0.9870603680610657 Batch_id=97 Accuracy=55.16: 100%|██████████| 98/98 [00:35<00:00,  2.74it/s]

Test set: Average loss: 0.0022, Accuracy: 6067/10000 (60.67%)

Validation accuracy increased (52.110000 --> 60.670000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 4
Learning rate = 0.068  for epoch:  4
Loss=0.9823508262634277 Batch_id=97 Accuracy=64.26: 100%|██████████| 98/98 [00:37<00:00,  2.58it/s]

Test set: Average loss: 0.0019, Accuracy: 6748/10000 (67.48%)

Validation accuracy increased (60.670000 --> 67.480000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 5
Learning rate = 0.06477894736842106  for epoch:  5
Loss=0.7789981365203857 Batch_id=97 Accuracy=71.10: 100%|██████████| 98/98 [00:41<00:00,  2.37it/s]

Test set: Average loss: 0.0018, Accuracy: 6845/10000 (68.45%)

Validation accuracy increased (67.480000 --> 68.450000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 6
Learning rate = 0.06155789473684211  for epoch:  6
Loss=0.7338870167732239 Batch_id=97 Accuracy=74.35: 100%|██████████| 98/98 [00:41<00:00,  2.37it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0019, Accuracy: 6576/10000 (65.76%)

EPOCH: 7
Learning rate = 0.05833684210526316  for epoch:  7
Loss=0.6311578154563904 Batch_id=97 Accuracy=76.51: 100%|██████████| 98/98 [00:41<00:00,  2.37it/s]

Test set: Average loss: 0.0012, Accuracy: 7909/10000 (79.09%)

Validation accuracy increased (68.450000 --> 79.090000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 8
Learning rate = 0.055115789473684215  for epoch:  8
Loss=0.6519821286201477 Batch_id=97 Accuracy=78.29: 100%|██████████| 98/98 [00:41<00:00,  2.33it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0015, Accuracy: 7436/10000 (74.36%)

EPOCH: 9
Learning rate = 0.05189473684210527  for epoch:  9
Loss=0.6197738647460938 Batch_id=97 Accuracy=78.96: 100%|██████████| 98/98 [00:43<00:00,  2.26it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0013, Accuracy: 7663/10000 (76.63%)

EPOCH: 10
Learning rate = 0.04867368421052632  for epoch:  10
Loss=0.5957424640655518 Batch_id=97 Accuracy=80.01: 100%|██████████| 98/98 [00:44<00:00,  2.20it/s] 
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0017, Accuracy: 7274/10000 (72.74%)

EPOCH: 11
Learning rate = 0.04545263157894737  for epoch:  11
Loss=0.6328437328338623 Batch_id=97 Accuracy=81.03: 100%|██████████| 98/98 [00:43<00:00,  2.26it/s] 
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0017, Accuracy: 7079/10000 (70.79%)

EPOCH: 12
Learning rate = 0.042231578947368426  for epoch:  12
Loss=0.6438823938369751 Batch_id=97 Accuracy=81.83: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s] 

Test set: Average loss: 0.0011, Accuracy: 8134/10000 (81.34%)

Validation accuracy increased (79.090000 --> 81.340000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 13
Learning rate = 0.03901052631578948  for epoch:  13
Loss=0.42896437644958496 Batch_id=97 Accuracy=82.56: 100%|██████████| 98/98 [00:41<00:00,  2.38it/s]

Test set: Average loss: 0.0010, Accuracy: 8325/10000 (83.25%)

Validation accuracy increased (81.340000 --> 83.250000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 14
Learning rate = 0.03578947368421053  for epoch:  14
Loss=0.4407457709312439 Batch_id=97 Accuracy=83.52: 100%|██████████| 98/98 [00:40<00:00,  2.43it/s] 
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0010, Accuracy: 8275/10000 (82.75%)

EPOCH: 15
Learning rate = 0.03256842105263158  for epoch:  15
Loss=0.4933714270591736 Batch_id=97 Accuracy=84.12: 100%|██████████| 98/98 [00:41<00:00,  2.34it/s] 
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0011, Accuracy: 8175/10000 (81.75%)

EPOCH: 16
Learning rate = 0.029347368421052636  for epoch:  16
Loss=0.48405155539512634 Batch_id=97 Accuracy=84.74: 100%|██████████| 98/98 [00:43<00:00,  2.25it/s]

Test set: Average loss: 0.0009, Accuracy: 8486/10000 (84.86%)

Validation accuracy increased (83.250000 --> 84.860000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 17
Learning rate = 0.026126315789473692  for epoch:  17
Loss=0.4573661684989929 Batch_id=97 Accuracy=85.70: 100%|██████████| 98/98 [00:43<00:00,  2.24it/s] 
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0010, Accuracy: 8308/10000 (83.08%)

EPOCH: 18
Learning rate = 0.02290526315789474  for epoch:  18
Loss=0.41120025515556335 Batch_id=97 Accuracy=86.65: 100%|██████████| 98/98 [00:40<00:00,  2.43it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.0009, Accuracy: 8401/10000 (84.01%)

EPOCH: 19
Learning rate = 0.01968421052631579  for epoch:  19
Loss=0.39725521206855774 Batch_id=97 Accuracy=87.40: 100%|██████████| 98/98 [00:42<00:00,  2.32it/s]

Test set: Average loss: 0.0009, Accuracy: 8493/10000 (84.93%)

Validation accuracy increased (84.860000 --> 84.930000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 20
Learning rate = 0.016463157894736846  for epoch:  20
Loss=0.3162410855293274 Batch_id=97 Accuracy=88.60: 100%|██████████| 98/98 [00:44<00:00,  2.21it/s] 

Test set: Average loss: 0.0007, Accuracy: 8816/10000 (88.16%)

Validation accuracy increased (84.930000 --> 88.160000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 21
Learning rate = 0.013242105263157902  for epoch:  21
Loss=0.28780344128608704 Batch_id=97 Accuracy=89.61: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]

Test set: Average loss: 0.0006, Accuracy: 8926/10000 (89.26%)

Validation accuracy increased (88.160000 --> 89.260000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 22
Learning rate = 0.010021052631578951  for epoch:  22
Loss=0.3368803560733795 Batch_id=97 Accuracy=90.66: 100%|██████████| 98/98 [00:43<00:00,  2.27it/s] 

Test set: Average loss: 0.0006, Accuracy: 9035/10000 (90.35%)

Validation accuracy increased (89.260000 --> 90.350000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 23
Learning rate = 0.0068000000000000005  for epoch:  23
Loss=0.19240239262580872 Batch_id=97 Accuracy=92.02: 100%|██████████| 98/98 [00:39<00:00,  2.50it/s]

Test set: Average loss: 0.0006, Accuracy: 9045/10000 (90.45%)

Validation accuracy increased (90.350000 --> 90.450000).  Saving model ...
 
