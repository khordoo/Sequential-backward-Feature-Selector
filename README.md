# Sequential-backward-Feature-Selector
Python implementation of Sequential backward Feature Selection from scratch.


* The input to the model is the normalized X with corresponding target y variable.
* 5-fold cross-validation was used for measuring accuracy.
* features are removed until we reach a desired set of reduced features .

The class provides a best_feature properties that keeps the index of best features during 
each reduced feature subspace. 

Here is an example of the best_features run on Wines dataset. :

```json
[
  {
    "featureSize": 13,
    "score": 0.9593333333333334,
    "features": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  },
  {
    "featureSize": 12,
    "score": 0.9673333333333334,
    "features": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
  },
  {
    "featureSize": 11,
    "score": 0.9676666666666666,
    "features": [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
  },
  {
    "featureSize": 10,
    "score": 0.9756666666666668,
    "features": [0, 2, 3, 4, 6, 7, 8, 9, 10, 12]
  },
  {
    "featureSize": 9,
    "score": 0.9756666666666668,
    "features": [0, 2, 3, 4, 6, 7, 8, 10, 12]
  },
  {
    "featureSize": 8,
    "score": 0.9753333333333334,
    "features": [0, 2, 3, 4, 6, 8, 10, 12]
  },
  { "featureSize": 7, "score": 0.992, "features": [0, 2, 3, 6, 8, 10, 12] },
  { "featureSize": 6, "score": 0.992, "features": [0, 2, 3, 6, 10, 12] },
  { "featureSize": 5, "score": 0.976, "features": [0, 2, 3, 6, 10] },
  { "featureSize": 4, "score": 0.9753333333333334, "features": [0, 2, 3, 6] },
  { "featureSize": 3, "score": 0.9513333333333331, "features": [0, 2, 6] },
  { "featureSize": 2, "score": 0.9183333333333333, "features": [0, 6] },
  { "featureSize": 1, "score": 0.8293333333333333, "features": [6] }
]

```

Here is the plot of accuracy vs the number of features:

![image](https://user-images.githubusercontent.com/32692718/81483705-5eb69980-91fd-11ea-9ed0-9c1e213e8662.png)

As you can see the accuracy  of our KNN classifier has increased, as we reduced the number of features in the training dataset. This is likely due to the decrease in the model overfitting by decreasing the curse of dimentionality.

From the plot we can see that 6 feature provides a good accuracy. Lets evaluate the perofmance of our classifier using our test dataset: 

Test accuracy on test data set with full features : 0.9444444444444444
Test Accuracy on test data set with reduced features (6): 0.9722222222222222
