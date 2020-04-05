# Machine Learning Final Assignment 2 

### Model Selection: Perceptron

## Major Changes to Sample Code

### Removal of example features
Due to the implementation of the Histogram of Oriented Gradients, the image was not downscaled and no other image manipluation was completed. 

### Histogram of Oriented Gradients
I would like to reference [this](https://gurus.pyimagesearch.com/lesson-sample-histogram-of-oriented-gradients-and-car-logo-recognition/) article for explaining clearly the different parameters of using the Histogram of Oriented Gradients descriptors.
Each paramter was testing individually with the majority of testing focused on ```orientations```, ```pixels_per_cell``` & ```cells_per_block```.
pixels_per_cell & cells_per_block work in conjunction with each other. Raising the pixels_per_cell value would be balanced by a lower cells_per_block and vice versa.
Orientation values were tested between 4 & 20. Parameter values were chosen based on their ability to improve both perceptron and neural network models.

```python
    final_image = feature.hog(img_raw, orientations=16, pixels_per_cell=(6, 7), cells_per_block=(4, 4), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=False)
```

### Model Opimization
Both the Perceptron and Neural Network Model performed extremely well with mininal changes from the defaults. Test results ranged from 'FP': 5, 'FN': 5 to 'FP': 0, 'FN': 2. 
Optimization of the ```penalty``` parameter, (testing l2, l1 & elasticnet) to elasticnet and the ```alpha``` value (testing from 0.00001 to 10) to 0.00001 for the Perceptron model lowered the test results to FP': 0, 'FN': 2, with some inconsistency to FP': 0, 'FN': 3.
```python
prc = linear_model.SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=0.00001)
```
Optimization of the neural network values saw similar results to the Perceptron with more variation in results on generally slightly lower results by either of 1 or 2 false negatives. 
```python
nn = neural_network.MLPClassifier(hidden_layer_sizes=(100,5), max_iter=1000)
```
Due to the lower variation in results the **perceptron** model was selected for the submission.


