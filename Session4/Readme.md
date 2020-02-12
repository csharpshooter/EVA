EVA 4 Session 4 assignment
------------------------------------------
Achieved 99.29% accuracy in 19th Epoch
------------------------------------------
Details
--------
1. Tried to use Batch normalization but was getting less and varying accuracy so did not use it.
2. Also tried dropout of 0.05 after every convolution, accuracy was not increasing much and also needed to train network more  to achieve same accuracy.
3. Wanted to implement GAP, was getting some errors related to 2d and 3d tensor, but was not able to give enough time to implement it correctly

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
            Conv2d-2           [-1, 16, 28, 28]           2,320
         MaxPool2d-3           [-1, 16, 14, 14]               0
            Conv2d-4           [-1, 16, 14, 14]           2,320
            Conv2d-5           [-1, 16, 14, 14]           2,320
         MaxPool2d-6             [-1, 16, 7, 7]               0
            Conv2d-7             [-1, 16, 5, 5]           2,320
            Conv2d-8             [-1, 32, 3, 3]           4,640
            Conv2d-9             [-1, 10, 1, 1]           2,890
================================================================
Total params: 16,970
Trainable params: 16,970
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.27
Params size (MB): 0.06
Estimated Total Size (MB): 0.34
----------------------------------------------------------------


Logs:
-----

 0%|          | 0/1875 [00:00<?, ?it/s]
Epoch:  1
loss=0.01633894443511963 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 183.37it/s]  
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0730, Accuracy: 9781/10000 (98%)

Learning rate = 0.015  for epoch:  2

Epoch:  2
loss=0.048518478870391846 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 183.03it/s]  
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0557, Accuracy: 9818/10000 (98%)

Learning rate = 0.015  for epoch:  3

Epoch:  3
loss=0.006362110376358032 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 182.41it/s]  
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0440, Accuracy: 9862/10000 (99%)

Learning rate = 0.015  for epoch:  4

Epoch:  4
loss=0.025394409894943237 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 181.99it/s]  
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0429, Accuracy: 9866/10000 (99%)

Learning rate = 0.015  for epoch:  5

Epoch:  5
loss=0.0026060640811920166 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 184.12it/s] 
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0386, Accuracy: 9869/10000 (99%)

Learning rate = 0.015  for epoch:  6

Epoch:  6
loss=0.005535602569580078 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 183.38it/s]  
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0422, Accuracy: 9873/10000 (99%)

Learning rate = 0.015  for epoch:  7

Epoch:  7
loss=0.0016115307807922363 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 183.31it/s] 
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0346, Accuracy: 9893/10000 (99%)

Learning rate = 0.0075  for epoch:  8

Epoch:  8
loss=0.037798553705215454 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 182.78it/s]  
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0280, Accuracy: 9906/10000 (99%)

Learning rate = 0.0075  for epoch:  9

Epoch:  9
loss=0.000371396541595459 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 180.45it/s]  
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0291, Accuracy: 9914/10000 (99%)

Learning rate = 0.0075  for epoch:  10

Epoch:  10
loss=0.00021719932556152344 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 182.23it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0332, Accuracy: 9904/10000 (99%)

Learning rate = 0.0075  for epoch:  11

Epoch:  11
loss=0.0017279982566833496 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 181.14it/s] 
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0406, Accuracy: 9894/10000 (99%)

Learning rate = 0.0075  for epoch:  12

Epoch:  12
loss=0.09103935956954956 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 180.77it/s]   
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0363, Accuracy: 9914/10000 (99%)

Learning rate = 0.0075  for epoch:  13

Epoch:  13
loss=0.0026869475841522217 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 177.29it/s] 
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0412, Accuracy: 9899/10000 (99%)

Learning rate = 0.0075  for epoch:  14

Epoch:  14
loss=0.0006703734397888184 batch_id=1874: 100%|██████████| 1875/1875 [00:09<00:00, 188.52it/s] 
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0387, Accuracy: 9915/10000 (99%)

Learning rate = 0.0075  for epoch:  15

Epoch:  15
loss=2.5928020477294922e-05 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 185.90it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0447, Accuracy: 9903/10000 (99%)

Learning rate = 0.00375  for epoch:  16

Epoch:  16
loss=1.817941665649414e-05 batch_id=1874: 100%|██████████| 1875/1875 [00:09<00:00, 188.75it/s] 
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0360, Accuracy: 9926/10000 (99%)

Learning rate = 0.00375  for epoch:  17

Epoch:  17
loss=3.129243850708008e-06 batch_id=1874: 100%|██████████| 1875/1875 [00:09<00:00, 189.22it/s] 
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0394, Accuracy: 9927/10000 (99%)

Learning rate = 0.00375  for epoch:  18

Epoch:  18
loss=5.960464477539063e-08 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 182.12it/s] 
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0407, Accuracy: 9926/10000 (99%)

Learning rate = 0.00375  for epoch:  19

Epoch:  19
loss=0.00016635656356811523 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 182.68it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0422, Accuracy: 9929/10000 (99%)

Learning rate = 0.00375  for epoch:  20

Epoch:  20
loss=1.1920928955078125e-07 batch_id=1874: 100%|██████████| 1875/1875 [00:10<00:00, 179.85it/s]
Test set: Average loss: 0.0428, Accuracy: 9925/10000 (99%)

Learning rate = 0.00375  for epoch:  21


