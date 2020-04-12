## EVA Assignment 11
----------------------
## Name : Abhijit Mali
----------------------
## Notes 
---------------------------------------------------------------------------------------------------------------------------
1. Wrote own function for One Cycle Policy -> src->train->trainhelper.py. Traingle plot was done with Learning Rate (y-axis) vs Epochs on X-Axis using matplot lib.

![triangleplot](https://github.com/csharpshooter/EVA/blob/master/A11/images/TrianglePlot.png)

2. Ran model for 25 epochs for lr range test. LR range tested from 0.001 to 0.096. Found optimal Lr between 0.06 to 0.096.
Max LR = 0.068 ,Min LR = Max LR / 13 = 0.00523

![lrrangefinder](https://github.com/csharpshooter/EVA/blob/master/A11/images/lrrangetestgraph.png)

3. Max train accuracy =  93.59, max test accuracy = 91.7.

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
 ![Graphs](https://github.com/csharpshooter/EVA/blob/master/A11/images/TensorBoardGraphs.png)
 ![Dist](https://github.com/csharpshooter/EVA/blob/master/A11/images/TensorBoardDistribution.png)
    
---------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------
## Project Structure
--------------------

![ProjectStructure](https://github.com/csharpshooter/EVA/blob/master/A11/images/projectstructure.png)

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
Learning rate = 0.00523  for epoch:  0
/home/abhijit/.virtualenvs/dl4cv/lib/python3.6/site-packages/torch/nn/_reduction.py:43: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))
Loss=1.7145285606384277 Batch_id=97 Accuracy=25.03: 100%|██████████| 98/98 [00:19<00:00,  4.98it/s]
Test set: Average loss: 0.0035, Accuracy: 3556/10000 (35.56%)

Validation accuracy increased (0.000000 --> 35.560000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 1
Learning rate = 0.020922500000000004  for epoch:  1
Loss=1.4757856130599976 Batch_id=97 Accuracy=34.77: 100%|██████████| 98/98 [00:19<00:00,  4.93it/s]
Test set: Average loss: 0.0031, Accuracy: 4613/10000 (46.13%)

Validation accuracy increased (35.560000 --> 46.130000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 2
Learning rate = 0.036615  for epoch:  2
Loss=1.3833483457565308 Batch_id=97 Accuracy=45.59: 100%|██████████| 98/98 [00:29<00:00,  3.31it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 4316/10000 (43.16%)

EPOCH: 3
Learning rate = 0.05230750000000001  for epoch:  3
Loss=1.1887569427490234 Batch_id=97 Accuracy=51.75: 100%|██████████| 98/98 [00:34<00:00,  2.86it/s]
Test set: Average loss: 0.0024, Accuracy: 5615/10000 (56.15%)

Validation accuracy increased (46.130000 --> 56.150000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 4
Learning rate = 0.068  for epoch:  4
Loss=1.2031654119491577 Batch_id=97 Accuracy=62.59: 100%|██████████| 98/98 [00:34<00:00,  2.81it/s]
Test set: Average loss: 0.0020, Accuracy: 6475/10000 (64.75%)

Validation accuracy increased (56.150000 --> 64.750000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 5
Learning rate = 0.06469631578947369  for epoch:  5
Loss=0.8196614384651184 Batch_id=97 Accuracy=69.39: 100%|██████████| 98/98 [00:36<00:00,  2.65it/s]
Test set: Average loss: 0.0017, Accuracy: 7027/10000 (70.27%)

Validation accuracy increased (64.750000 --> 70.270000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 6
Learning rate = 0.06139263157894737  for epoch:  6
Loss=0.7618541717529297 Batch_id=97 Accuracy=73.83: 100%|██████████| 98/98 [00:40<00:00,  2.42it/s]
Test set: Average loss: 0.0013, Accuracy: 7726/10000 (77.26%)

Validation accuracy increased (70.270000 --> 77.260000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 7
Learning rate = 0.05808894736842106  for epoch:  7
Loss=0.6519532203674316 Batch_id=97 Accuracy=76.82: 100%|██████████| 98/98 [00:39<00:00,  2.49it/s]
Test set: Average loss: 0.0012, Accuracy: 7893/10000 (78.93%)

Validation accuracy increased (77.260000 --> 78.930000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 8
Learning rate = 0.05478526315789474  for epoch:  8
Loss=0.624919056892395 Batch_id=97 Accuracy=78.64: 100%|██████████| 98/98 [00:41<00:00,  2.36it/s] 
Test set: Average loss: 0.0011, Accuracy: 8154/10000 (81.54%)

Validation accuracy increased (78.930000 --> 81.540000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 9
Learning rate = 0.05148157894736842  for epoch:  9
Loss=0.5436624884605408 Batch_id=97 Accuracy=80.40: 100%|██████████| 98/98 [00:39<00:00,  2.46it/s] 
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.0013, Accuracy: 7731/10000 (77.31%)

EPOCH: 10
Learning rate = 0.04817789473684211  for epoch:  10
Loss=0.539050817489624 Batch_id=97 Accuracy=81.63: 100%|██████████| 98/98 [00:42<00:00,  2.32it/s]  
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.0012, Accuracy: 7817/10000 (78.17%)

EPOCH: 11
Learning rate = 0.044874210526315794  for epoch:  11
Loss=0.537773847579956 Batch_id=97 Accuracy=82.66: 100%|██████████| 98/98 [00:40<00:00,  2.41it/s]  
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.0016, Accuracy: 7272/10000 (72.72%)

EPOCH: 12
Learning rate = 0.041570526315789474  for epoch:  12
Loss=0.5607398152351379 Batch_id=97 Accuracy=83.37: 100%|██████████| 98/98 [00:40<00:00,  2.40it/s] 
Test set: Average loss: 0.0011, Accuracy: 8162/10000 (81.62%)

Validation accuracy increased (81.540000 --> 81.620000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 13
Learning rate = 0.038266842105263155  for epoch:  13
Loss=0.41175684332847595 Batch_id=97 Accuracy=84.22: 100%|██████████| 98/98 [00:41<00:00,  2.38it/s]
Test set: Average loss: 0.0010, Accuracy: 8299/10000 (82.99%)

Validation accuracy increased (81.620000 --> 82.990000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 14
Learning rate = 0.03496315789473684  for epoch:  14
Loss=0.4432563781738281 Batch_id=97 Accuracy=85.17: 100%|██████████| 98/98 [00:40<00:00,  2.39it/s] 
Test set: Average loss: 0.0008, Accuracy: 8646/10000 (86.46%)

Validation accuracy increased (82.990000 --> 86.460000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 15
Learning rate = 0.03165947368421053  for epoch:  15
Loss=0.39080023765563965 Batch_id=97 Accuracy=86.00: 100%|██████████| 98/98 [00:40<00:00,  2.41it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.0009, Accuracy: 8414/10000 (84.14%)

EPOCH: 16
Learning rate = 0.02835578947368421  for epoch:  16
Loss=0.41148507595062256 Batch_id=97 Accuracy=86.95: 100%|██████████| 98/98 [00:45<00:00,  2.15it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.0008, Accuracy: 8617/10000 (86.17%)

EPOCH: 17
Learning rate = 0.02505210526315789  for epoch:  17
Loss=0.3731699287891388 Batch_id=97 Accuracy=87.74: 100%|██████████| 98/98 [00:41<00:00,  2.35it/s] 
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.0008, Accuracy: 8644/10000 (86.44%)

EPOCH: 18
Learning rate = 0.021748421052631577  for epoch:  18
Loss=0.3474939465522766 Batch_id=97 Accuracy=88.82: 100%|██████████| 98/98 [00:38<00:00,  2.53it/s] 
Test set: Average loss: 0.0008, Accuracy: 8658/10000 (86.58%)

Validation accuracy increased (86.460000 --> 86.580000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 19
Learning rate = 0.018444736842105264  for epoch:  19
Loss=0.3990764617919922 Batch_id=97 Accuracy=89.27: 100%|██████████| 98/98 [00:42<00:00,  2.31it/s] 
Test set: Average loss: 0.0008, Accuracy: 8659/10000 (86.59%)

Validation accuracy increased (86.580000 --> 86.590000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 20
Learning rate = 0.015141052631578944  for epoch:  20
Loss=0.2886563837528229 Batch_id=97 Accuracy=90.16: 100%|██████████| 98/98 [00:42<00:00,  2.32it/s] 
Test set: Average loss: 0.0007, Accuracy: 8905/10000 (89.05%)

Validation accuracy increased (86.590000 --> 89.050000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 21
Learning rate = 0.011837368421052624  for epoch:  21
Loss=0.24585449695587158 Batch_id=97 Accuracy=91.38: 100%|██████████| 98/98 [00:38<00:00,  2.57it/s]
Test set: Average loss: 0.0006, Accuracy: 9028/10000 (90.28%)

Validation accuracy increased (89.050000 --> 90.280000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 22
Learning rate = 0.008533684210526311  for epoch:  22
Loss=0.26173126697540283 Batch_id=97 Accuracy=92.41: 100%|██████████| 98/98 [00:43<00:00,  2.25it/s]
Test set: Average loss: 0.0006, Accuracy: 9058/10000 (90.58%)

Validation accuracy increased (90.280000 --> 90.580000).  Saving model ...
  0%|          | 0/98 [00:00<?, ?it/s]
EPOCH: 23
Learning rate = 0.0052299999999999985  for epoch:  23
Loss=0.15184636414051056 Batch_id=97 Accuracy=93.59: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]
Test set: Average loss: 0.0005, Accuracy: 9170/10000 (91.70%)

Validation accuracy increased (90.580000 --> 91.700000).  Saving model ...
