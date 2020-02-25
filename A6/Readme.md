EVA Assignment 6
----------------
Name : Abhijit Mali
-------------------

L1 and L2 accuracy and loss change Graph:
-----------------------------------------
![L1 and L2 accuracy and loss change graph](https://github.com/csharpshooter/EVA/blob/master/A6/L1L2RegularizationComparisonGraph.png)


Observation w.r.t. L1 and L2's performance in the regularization of model
-------------------------------------------------------------------------
1) Had to give very low value for L1 regularisation so that model would train properly also achieved stable accuracy with L1
but model test accuracy reduced. set factor = 0.00005 for L1
2) set weight decay = weight_decay=0.001 for L2. Was able to achieved good test accuracy with L2 but accuracu was varying a bit
3) With both L1 and L2 got good performance but was not able to train model for higher test and train accuracy.
