This chapter covered the workflow of Machine Learning in detail. In summary, they are as follows.

# Defining the problem and assembling a dataset
What will your input data be? What are you trying to predict? What *type* of problem are you facing? Is it binary classification? Multiclass classification? Scalar regression?

You can’t move to the next stage until you know what your inputs and outputs are, and what data you’ll use. Be aware of the hypotheses you make at this stage:
- You hypothesize that your outputs can be predicted given your inputs.
- You hypothesize that your available data is sufficiently informative to learn the relationship between inputs and outputs.

**Not all problems can be solved; just because you’ve assembled examples of inputs X and targets Y doesn’t mean X contains enough information to predict Y.**

# Choosing a measure of success
To achieve success, you must define what you mean by success—accuracy? Precision and recall? Customer-retention rate? Your metric for success will guide the choice of a loss function: what your model will optimize. It should directly align with your higher-level goals, such as the success of your business.

For class-imbalanced problems, you can use precision and recall. For ranking problems or multilabel classification, you can use mean average precision. And it isn’t uncommon to have to define your own custom metric by which to measure success.

# Deciding on an evaluation protocol
We’ve previously reviewed three common evaluation protocols:
- **Maintaining a hold-out validation set**—The way to go when you have plenty of data
- **Doing K-fold cross-validation**—The right choice when you have too few samples for hold-out validation to be reliable
- **Doing iterated K-fold validation**—For performing highly accurate model evaluation when little data is available

# Preparing your data
- As you saw previously, your data should be formatted as tensors.
- The values taken by these tensors should usually be scaled to small values: for example, in the [-1, 1] range or [0, 1] range.
- If different features take values in different ranges (heterogeneous data), *then the data should be normalized.*
- You may want to do some feature engineering, especially for small-data problems.

# Developing a model that does better than a baseline
Your goal at this stage is to achieve *statistical power*: that is, to develop a small model that is capable of beating a dumb baseline.

Note that it’s not always possible to achieve statistical power. **If you can’t beat a random baseline after trying multiple reasonable architectures, it may be that the answer to the question you’re asking isn’t present in the input data.**

Assuming that things go well, you need to make three key choices to build your first working model:

- **Last-layer activation**—This establishes useful constraints on the network’s output.
- **Loss function**—This should match the type of problem you’re trying to solve.
- **Optimization configuration**—What optimizer will you use? What will its learning rate be? In most cases, it’s safe to go with rmsprop and its default learning rate.


Regarding the choice of a loss function, note that it isn’t always possible to directly optimize for the metric that measures success on a problem. Sometimes there is no easy way to turn a metric into a loss function; loss functions, after all, need to be computable given only a mini-batch of data and must be differentiable.

The table below can help you choose a last-layer activation and a loss function for a few common problem types.

|             **Problem type**            | **Last-layer activation** |        **Loss function**       |
|:---------------------------------------:|:-------------------------:|:------------------------------:|
| Binary classification                   |          sigmoid          |      `binary_crossentropy`     |
| Multiclass, single-label classification |          softmax          |   `categorical_crossentropy`   |
| Multiclass, multilabel classification   |          sigmoid          |      `binary_crossentropy`     |
| Regression to arbitrary values          |            None           |              `mse`             |
| Regression to values between 0 and 1    |          sigmoid          | `mse` or `binary_crossentropy` |

# Scaling up: developing a model that overfits
Once you’ve obtained a model that has statistical power, the question becomes, is your model sufficiently powerful? Does it have enough layers and parameters to properly model the problem at hand?

Remember that the universal tension in machine learning is between optimization and generalization; the ideal model is one that stands right at the border between underfitting and overfitting; between undercapacity and overcapacity. To figure out where this border lies, first you must cross it.

To figure out how big a model you’ll need, you must develop a model that overfits.  
This is fairly easy:

1. Add layers.
2. Make the layers bigger.
3. Train for more epochs.

Always monitor the training loss and validation loss, as well as the training and validation values for any metrics you care about. When you see that the model’s performance on the validation data begins to degrade, you’ve achieved overfitting.

# Regularizing your model and tuning your hyperparameters
This step will take the most time: you’ll repeatedly modify your model, train it, evaluate on your validation data (not the test data, at this point), modify it again, and repeat, until the model is as good as it can get. These are some things you should try:

- Add dropout
- Try different architectures: add or remove layers.
- Add L1 and/or L2 regularization.
- Try different hyperparameters (such as the number of units per layer or the learning rate of the optimizer) to find the optimal configuration.
- Optionally, iterate on feature engineering: add new features, or remove features that don’t seem to be informative.

Be mindful of the following: every time you use feedback from your validation process to tune your model, you leak information about the validation process into the model. Once you’ve developed a satisfactory model configuration, you can train your final production model on all the available data (training and validation) and evaluate it one last time on the test set.

If it turns out that performance on the test set is significantly worse than the performance measured on the validation data, this may mean either that your validation procedure wasn’t reliable after all, or that you began over-fitting to the validation data while tuning the parameters of the model. In this case, you may want to switch to a more reliable evaluation protocol (such as iterated K-fold validation).