# Preprocessing
You will usually need to preprocess raw data before feeding it into a neural network, creating tensors of appropriate shapes for the problem at hand:
- **Vector data**-- 2D tensors of shape `(samples, features)`
- **Timeseries data or sequence data**-- 3D tensors of shape `(samples,timesteps, features)`
- **Images**-- 4D tensors of shape `(samples, height, width, channels)` or `(samples, channels, height, width)`
- **Video**-- 5D tensors of shape `(samples, frames, height, width, channels)` or `(samples, frames, channels, height, width)`

## Scaling
When your data has features with different ranges, scale each feature independently as part of preprocessing.

# Training
As training progresses, neural networks eventually begin to overfit and obtain worse results on never-before-seen data.

If you don't have much training data, use a small network with only one or two hidden layers, to avoid severe overfitting.

If your data is divided into many categories, you may cause information bottlenecks if you make the intermediate layers too small.


# Loss functions
Regression uses different loss functions and different evaluation metrics than classification.
`binary_crossentropy` is ideal for classification problems with only two labels.
`categorical_crossentropy` is ideal for classification problems with multiple classification labels.
`mse` or 'mean squared error' is better for regression problems than classification.


# K-Fold Cross Validation
When you’re working with little data, K-fold validation can help reliably evaluate your model.
We performed K-fold cross validation in the Boston dataset example.