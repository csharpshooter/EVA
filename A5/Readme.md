**EVA 4 Session 4 Assignment**
------------------------------

**Name: Abhijit Mali**
----------------------

-----------------------------------------------------------------------------------------------------------------------------------------

Step 1:
-------
Initial Run of base model: No change in model Ran base given model for 15 epochs

Observations for initial run: Best training accuracy : 99.96 Best test accuracy : 99.27 No of params : 6,379,786

Conculsions for initial run: 1) As training accuracy is very nearly 100% and test – train accuracy = 0.69. So Model is overfitting but will still not add dropout as have to reduce no of parameter as target parameters are 10,000 or less

**Target:** Reduce params to less than 10k

**For Step 1 changes made to base model:** 1) Changed kernel size in model to make model have less than 10000 params Used kernel of 10,15 features to get 8,340 params for our model.

**Results:** Best train accuracy = 99.18 Best test accuracy = 98.94 Parameters = 8,340

**Analysis:** Model can be trained further for as train acc is not near to 100% and by doing so we can further improve our test accuracy as well . Diff between train and test accuracy = 0.24

-----------------------------------------------------------------------------------------------------------------------------------------

Step 2:
-------
**Target for step 2:** Train model further to improve accuracy

**Changes made:** Add batch normalisation to each conv2d layer except last layer and train model further to give better effciency

**Result:** Best train accuracy = 99.71 Best test accuracy = 99.32 Parameters = 8,480

**Analysis:** After adding batch normalisation train accuracy improved by 0.53 from previous model output and test accuracy improved by 0.38. So after adding batch normalisation it is seen that train and test accuracy have increased nearly the same and there is still scope of training model to 100 percent and model can be improved further.

-----------------------------------------------------------------------------------------------------------------------------------------

Step 3:
-------
**Target for step 3:** Train model further try to acheive 99.40 target as in step 2 we were near to target got 99.32 test accuracy

**Changes made:** Add gap and remove last big kernel

**Result:** Best train accuracy = 99.14 Best test accuracy = 99.15 Parameters = 6,250

**Analysis:** As last big kernel was removed the no of paramters has decreased from 8480 to 6,250 hence accuracy has decreased by nearly more than .50 Model can be trained further by adding more layers before gap. Also looking at the test loss and test accuracy graph we can see that the graphs are varying in the accurracy and loss between the epochs. Also seen in most of the epochs that test accuracy is more that the train accuracy that mean model is underfitting. The model is not learning correctly and might predict wrong output. Underfitting can occur due to over regularization. It might be due to batch normalization and the last big kernel layer we removed resulting in over regularization as batch normalization is a form of regularization

-----------------------------------------------------------------------------------------------------------------------------------------

Step 4:
-------
**Target for step 4:** Train model further try to acheive 99.40 target as in step 3 we were near to target got 99.15 test accuracy

**Changes made :** Increase model capacity. Add more layers at the end i.e. before GAP layer

**Result:** Best train accuracy = 99.67 Best test accuracy = 99.47 Parameters = 9,755

**Analysis:** Target achieved in 4th step. Achieved target of 99.40 accuracy, got 99.4 or more accuracy in 6 epochs out of 15 epochs. Looking at the graphs and epochs output still feel that train and test accuracy is varying still and underfitting is still occuring a little.

----------------------------------------------------------------------------------------------------------------------------------------
