# HW3 Beras
:::info
Assignment due **October 28th at 6 pm EST** on gradescope
:::

## Assignment Overview

In this assignment you will begin constructing a basic Keras mimic, 🐻 Beras 🐻.

### Assignment Goals

1. Implement a simple Multi Layer Perceptron (MLP) model that mimics the Tensorflow/Keras API.
   - Implement core classes and methods used for **Auto Differentiation**.
   - Implement a **Dense Layer** similar to keras'.
   - Implement basic **preprocessing techniques** for use on the **MNIST Dataset**.
   - Implement a basic objective (loss) function for regression such as **MSE**.
   - Implement basic regression **accuracy metrics**.
   - **Learn** optimal weight and bias parameters using **gradient descent** and **backpropogation**.

2. Apply this model to predict digits using the MNIST Dataset

:::warning
__You should know:__ This is the longest assignment, we **highly** recommend starting early and reading this document carefully as you implement each part. 
:::
## Getting Started

### Stencil

<!--LINK THE REFERENCES-->

Please click [here](https://classroom.github.com/a/c-NcG5uQ) to get the stencil code. 

:::danger  
**Do not change the stencil except where specified.** You are welcome to write your own helper functions, however, changing the stencil's method signatures **will** break the autograder
:::

### Environment

You will need to use the virtual environment that you made in Homework 1. You can activate the environment by using the command `conda activate csci2470`. If you have any issues running the stencil code, be sure that your conda environment contains at least the following packages:

- `python==3.11`
- `numpy`
- `tensorflow==2.15`
- `pytest`

On Windows conda prompt or Mac terminal, you can check to see if a package is installed with:

```bash
conda list -n csci2470 <package_name>
```

On Unix systems to check to see if a package is installed you can use:

```bash
conda list -n csci2470 | grep <package_name>
```

:::danger
Be sure to read this handout in its **entirety before** moving onto implementing **any** part of the assignment!
:::
## Deep Learning Libraries

Deep learning is a very complicated and mathematically rich subject. However, when building models, all of these nuances can be abstracted away from the programmer through the use of deep learning libraries.

In this assignment you will be writting your own Deep Learning library, 🐻 Beras 🐻. You'll build everything you need to train a model on the MNIST dataset. The MNIST data contains 60k 28x28 black and white hand written digits, your model's job will be to classify which digit is in each image.  

:::danger
Please keep in mind you are _not_ allowed to use **_any_ Tensorflow, Keras, or PyTorch functions throughout HW3**. The autograder will intentionally not execute if you import these libraries.

:::

You are already familar with **Tensorflow** from our first assignment. Now your job will be to build your own version of it: Beras. 


## Roadmap

<!-- TODO: Finish writing this section -->

Don't worry if these tasks seem daunting at first glance! We've included a lot more info down below on specific implementation details.
1. Start with **`beras/core.py`** which will create some of the basic building blocks for the assignment. [Specifics](#1-berascorepy)
2. Move on to **`beras/layers.py`** to construct your own `Dense` layer. [Specifics](#2-beraslayerspy)
3. Now complete **`beras/activations.py`** [Specifics](#3-activations)
4. Continue with **`beras/losses.py`** to write **CategoricalCrossEntropy**. [Specifics](#4-beraslossespy)
5. Next write **CategoricalAccuracy** in **`beras/metrics.py`**. [Specifics](#5-berasmetricspy)
6. Implement **`beras/onehot.py`** which will be used later in preprocessing. [Specifics](#6-berasonehotpy)
7. Fill in the optimizer classes in **`beras/optimizer.py`**. [Specifics](#7-berasoptimizerpy)
8. Write **GradientTape** in **`beras/gradient_tape.py`**. [Specifics](#8-berasgradient_tapepy)
9. Construct the **Model** class in **`beras/model.py`**. [Specifics](#9-berasmodelpy)

:::danger
**GradientTape** is known to be tricky, so budget some extra time to implement it.
:::
10. Now you have finished the beras framework! Put it to use by implementing **`preprocessing.py`** to load and clean your data. [Specifics](#10-preprocessingpy)
11. Finally, write **`assignment.py`** to train a model on the MNIST Dataset! [Specifics](#11-assignmentpy)

Gaurav (DL TA 2024) put together this nice graphic to visualize the roadmap and how it all fits together. It's helpful to refer to as you go through the assignment!

![BERAS GRAPH (1)](https://hackmd.io/_uploads/HJ9yRMzk1e.png)
*Thanks Gaurav!*
## 1. beras/core.py
In this section we are going to prepare some abstract classes we will use for everything else we do in this assigment. This is a very important section since we will build everything else on top of this foundation.
:::info 
**Task 1.1 [Tensor]:** We will begin completing the construction of the `Tensor` class at the top of the file. Note that it subclasses the np.ndarray datatype, you can find out more about what that means <ins>[here](https://numpy.org/doc/stable/user/basics.subclassing.html)</ins>.

The only TODO is to pass in the data to the `a` kwarg in `np.asarray(a=???)` in the `__new__` method.
:::

:::warning
__You should know:__ You'll notice the `Tensor` class is nothing more than a standard `np.ndarray` but with an additional `trainable` attribute.

In Tensorflow there is also `tf.Variable` which you may see throughout the course. `tf.Variable` is a subclass of `tf.Tensor` but with some additional bells and whistles for convenience. Both are fair game for use when working with Tensorflow in this course.
:::

:::info
**Task 1.2 [Callable]:** There are no TODOs in `Callable` but it is important to familiarize yourself with this class. `Callable` simply allows its subclasses to use `self()` and `self.forward()` interchangably. More importantly, if a class subclasses `Callable` it **will** have a `forward` method that returns a `Tensor` and so, we **can and will** use these subclasses when **constructing layers and models** later. 
:::
:::warning
__You should know:__ __Keras__ and __Tensorflow__ use `call` instead of `forward` as the method name for the forward pass of a layer. __Pytorch__ and __Beras__ use `forward` to make the distinction between `__call__` and `forward` clear. 
:::

:::info
**Task 1.3 [Weighted]:** There are 4 methods in `Weighted` for you to fill out: `trainable_variables`, `non_trainable_variables`, `trainable`, `trainable (setter)`. Each method has a description and return type (if needed) in the stencil code. Be sure to follow the typing **exactly** or it's unlikely to pass the autograder.
:::

:::success
__Note:__ If you need a refreshing on python attributes and properties you can refer to <ins>[this](https://realpython.com/python-getter-setter/)</ins> helpful guide
:::
:::info
**Task 1.4 [Diffable.\_\_call__]** There are no coding TODOs in `Diffable.__call__` but it is **critical** that you spend some time to familize yourself with what it is doing. Understanding this method will help clear up later parts of the assignment.
:::
:::warning
__You should know:__ Recall that in python, `generic_class_name()` is equal to `generic_class_name.__call__()`. Note that Diffable implements the `__call__` method and __not__ the `forward` method. 

When we subclass `Diffable`, for example with `Dense`, we __will__ implement `forward` there. Then, when we use something like `dense_layer(inputs)` __the gradients will be recorded using `GradientTape`__ as you see in `Diffable.__call__`. If you use `dense_layer.forward(inputs)` __it will not record the gradients__ because `forward` won't handle the necessary logic.
:::

Finally, you will see the methods `compose_input_gradients` and `compose_weight_gradients`. These methods are responsible for composing the downstream gradient for during backpropagation using the upstream gradient, `J`, and the local input and wieght gradients of a `Diffable`. You don't have to give these methods a close read, but it's important to come back and use them later on. 

## 2. beras/layers.py
In this section we need to fill out the methods for `Dense`. We give you `__init__` and `weights`, you should read through both of these one liners to know what they are doing. PLease don't change these since the autograder relies on the naming conventions. Your tasks will be to implement the rest of the methods we need. 

:::info
__Task 1 [Dense.forward]:__  To begin, fill in the `Dense.forward` method. The parameter `x` represents our input. Remember from class that our Dense layer performs the following to get its output:

$$
f(\bf{x}) = \bf{x}\bf{W} + \bf{b}
$$

Keep in mind that `x` has shape `(num_samples, input_size)`.
:::
:::info
__Task 2 [Dense.get_input_gradients]:__ Refer to the formula you wrote in `Dense.forward` to compute $\frac{\partial f}{\partial x}$. Be sure to return the gradient `Tensor` __as a list__, this will come in handy when you write back propagation in `beras/gradient_tape.py` later in this assignment. 
:::
:::success
__Note:__ For each `Diffable` you can access the inputs of the forward method with `self.inputs`
:::

:::info
__Task 3 [Dense.get_weight_gradients]:__ Compute both $\frac{\partial f}{\partial w}$ and $\frac{\partial f}{\partial b}$ and return both `Tensor`s __in a list__, like you did in `Dense.get_input_gradients`.
:::

:::info
__Task 4 [Dense.\_initialize_weight]:__ Initialize the dense layer’s weight values. By default, return 0 for all weights (usually a bad idea). You are also required to allow for more sophisticated options by allowing for the following:
- **Normal:** Passing `normal` causes the weights to be initialized with a unit normal distribution $\mathcal{N}(0,1)$.
- **Xavier Normal:** Passing `xavier` causes the weights to be initialized in the same way as `keras.GlorotNormal`.
- **Kaiming He Normal:** Passing `kaiming` causes the weights to be initialized in the same way as `keras.HeNormal`.

Explicit definitions for each of these initializers can be found **[in the tensorflow docs](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)**
:::

:::warning
Notes: 
1. _initialize_weight __returns__ the weights and biases and does not set the weight attributes directly. 
2. Your weights should be `Variable` **not** `Tensor`. This is so that we can use the `.assign` method in your optimizers.
:::

## 3. beras/activations.py
Here, we will implement a couple activation functions that we will use when constructing our model. Here is some helpful information about the [activation functions](https://medium.com/analytics-vidhya/activation-functions-all-you-need-to-know-355a850d025e).

:::info
__Task 3.1 [LeakyReLU]:__ Fill out the forward pass and input gradients computation for `LeakyReLU`. You'll notice these are the same methods we implemented in `layers.py`, this is by design.
:::

:::success
__Hint:__ LeakyReLU is not continous so when computing the gradient, consider both the positive and negative cases. 

Note: Though LeakyReLU is technically not differentiable at $0$ exactly, we can just leave the gradient as $0$ for any $0$ input.
:::

:::info
__Task 3.2 [Sigmoid]:__ Complete the `forward` and `get_input_gradients` methods for `Sigmoid`.
:::

:::info
__Task 3.3 [Softmax]:__ Write the forward pass and gradient computation w.r.t inputs for Softmax. 
:::
:::success
__Hints:__
You should use stable softmax to prevent overflow and underflow issues. Details in the stencil.

Combining `np.outer` and `np.fill_diagonal` will significantly clean up the gradient computation

When you first try to compute the gradient it will become apparent that the input gradients are tricky. This [medium article](https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1) has a fantastic derivation that will make your life a lot easier.
:::

## 4. beras/losses.py
In this section we need to construct our Loss functions for the assignment, `MeanSquaredError` and `CategoricalCrossEntropy`. You should note that for most classification tasks we use `CategoricalCrossEntropy` by default but, for this assignment we will use both and compare the results. 

:::success
__Note:__ You'll notice we construct a `Loss` class that both `MeanSquaredError` and  `CategoricalCrossEntropy` inherit from. This is just so that we don't have to specify that our Loss functions don't have weights everytime we create one. 
:::
:::info
__Task 1 [MeanSquaredError.forward]:__ Implement the forward pass for `MeanSquaredError`. We want `(y_pred - y_true)**2`, and not the other way around. Don't forget that we expect to take in _batches_ of examples at a time so we will need to take the mean over the batch as well as the mean for each individual example. In short, the output should be the mean of means.  

Don't forget that `Tensors` are a subclass of `np.ndarrays` so we can use numpy methods!
:::
:::warning
__You should know:__ In general, loss functions should return exactly 1 scalar value no matter how many examples are in the batch. We take the mean loss from the batch examples in most practical cases. We will see later in the course that we can use multiple measures of loss at one time to train a model, in which case we often take a weighted sum of each individual loss as our loss value to backpropagate on. 
:::

:::info
__Task 2 [MeanSquaredError.get_input_gradients]:__ Just as we did for our dense layer, compute the gradient with respect to inputs in MeanSquaredError. It's important to rememeber that there are two inputs, `y_pred` and `y_true`. Since `y_true` comes from our database and is not dependent on our params, you should treat it like a constant vector. On the other hand, compute the gradient with respect to `y_pred` exactly as you did in `Dense`. Remember to return them both as a list!
:::
:::success
__Hint:__ If you aren't quite sure how to access your inputs, remember that `MeanSquaredError` is a `Diffable`!
:::

:::info
__Task 3 [CategoricalCrossEntropy.forward]:__ Implement the forward pass of `CategoricalCrossEntropy`. Make sure to find the per-sample average of the CCE Loss! You may run into trouble with values very close to 0 or 1, you may find `np.clip` of use...
:::

:::info
__Task 4 [CategoricalCrossEntropy.get_input_gradients]:__ Get input gradients for `CategoricalCrossEntropy`.
:::
## 5. beras/metrics.py
There isn't much to do in this file, just to implement the forward method for `CategoricalAccuracy`.

:::info
__Task 1 [CategoricalAccuracy.forward]:__ Fill in the `forward` method. Note that our input `probs` represents the probability of each class as predicted by the model and labels is a one hot encoded vector representing the true model class.
:::
:::success
__Hint:__ It may be helpful this also think of the labels as a probability distribution, where the probability of the true class is 1 and all other classes is 0. 

If the index of the max value in both vectors is the same, then our model has made the correct classification.
:::

## 6. beras/onehot.py
`onehot.py` only contains the `OneHotEncoder` class which is where you will code a one hot encoder to use on the data when you preprocess later in the assignment. Recall that a one hot encoder transforms a given value into a vector with all entries being 0 except one with a value of 1 (hence "one hot"). This is used often when we have mutliple discrete classes, like digits for the MNIST dataset.

:::info
__Task 1 [OneHotEncoder.fit]:__ In HW2 you were able to use `tf.onehot` now you get to build it yourself! In `OneHotEncoder.fit` you will take in a 1d vector of labels and you should construct a dictionary that maps each unique label to a one hot vector. This method doesn't return anything.

__Note__: you should only associate a one hot vector to labels present in labels! 
:::
:::success
__Hint:__ `np.unique` and `np.eye` may be of use here.
:::

:::info
__Task 2 [OneHotEncoder.forward]:__ Fill in the `OneHotEncoder.forward` method to transform the given 1d array `data` into a one-hot-encoded version of the data. This method should return a 2d `np.ndarray`.
:::

:::info
__Task 3 [OneHotEncoder.inverse]:__ `OneHotEncoder.inverse` should be an exact inverse of `OneHotEncoder.forward` such that `OneHotEncoder.inverse(OneHotEncoder.forward(data)) = data`.
:::


## 7. beras/optimizer.py
In `beras/optimizer.py` there are 3 optimizers we'd like you to implement, a `BasicOptimizer`, `RMSProp`, and `Adam`. In practice, `Adam` is tough to beat so more often than not you will default to using `Adam`. 

Each has an `__init__` and `apply_gradients`. We give you the `__init__` for each optimizer which contains all the hyperparams and variables you will need for each algorithm. Then in `apply_gradients` you will write the algorithm for each method to update the `trainable_params` according to the given `grads`. Both `trainable_params` and `grads` are lists with

$$\text{grad}[i] = \frac{\partial \mathcal{L}}{\partial \text{ trainable_params[i]}}$$

where $\mathcal{L}$ is the Loss of the network.

:::warning
**Note**: Make sure to use `param.assign` when updating the weights!
:::

:::info
__Task 1 [BasicOptimizer]:__ Write the `apply_gradients` method for the `BasicOptimizer`. 

For any given `trainable_param`, $w[i]$, and `learning_rate`, $r$, the optimization formula is given by
$$w[i] = w[i] - \frac{\partial \mathcal{L}}{\partial w[i]}*r$$
:::

:::info
__Task 2 [RMSProp]:__ Write the `apply_gradients` method for the `RMSProp`.

In `RMSProp` there are two new hyperparams, $\beta$ and $\epsilon$. 

$\beta$ is referred to as the __decay rate__ and typically defaults to .9. This decay rate has the effect of _lowering the learning rate as the model trains_. Intuitively, as our loss decreases we are closer to a minimum and should take smaller steps towards optimization to ensure we don't optimize past the minimum. 

$\epsilon$ is a small constant to prevent division by 0.

In addition to our hyperparams there is another term which we will call, __v__, which acts as the moving average of the gradients __for each param__. We update this value in addition to the `trainable_params` every time we apply the gradients.

For any given `trainable_param`, $w[i]$, `learning_rate`, $r$ the update is defined by
$$v[i] = \beta*v[i] + (1-\beta)*\left(\frac{\partial \mathcal{L}}{\partial w[i]}\right)^2$$
$$w[i] = w[i] - \frac{r}{\sqrt{v[i]} + \epsilon}*\frac{\partial \mathcal{L}}{\partial w[i]}$$

**Hint**: In our stencil code, we provide **v** as a dictionary which maps a key to a float. Keep in mind that we only need to store a single **v** value for each weight!
:::

:::info
__Task 3 [Adam]:__ Write the `apply_gradients` methods for the `Adam`.

At it's core Adam, is similar to `RMSProp` but it has more smoothing terms and computes an additional _momentum_ term to further balance the learning rate as we train. This momentum term has it's own decay term, $\beta_1$. Additionally, `Adam` keeps track of the number of optimization steps performed to further tweak the effective learning rate.

Here is what an optimization step with `Adam` looks like for `trainable_param`, $w[i]$ and `learning_rate`, $r$.

$$m[i] = m[i]*\beta_1 + (1-\beta_1)*\left(\frac{\partial \mathcal{L}}{\partial w[i]}\right)$$
$$v[i] = v[i]*\beta_2 + (1-\beta_2)*\left(\frac{\partial \mathcal{L}}{\partial w[i]}\right)^2$$
$$\hat{m} = m[i]/(1-\beta_1^t)$$
$$\hat{v} = v[i]/(1-\beta_2^t)$$
$$w[i] = w[i] - \frac{r*\hat{m}}{\sqrt{\hat{v}}+\epsilon}$$

Note: Don't forget the __iterate time once__ when `apply_gradients` is called!
:::

:::success
__Hint:__ Don't overcomplicate this section, it really is as simple as programming the algorithms as they are written.
:::

## 8. beras/gradient_tape.py
In `beras/gradient_tape` you are going to implement your very own context manager `GradientTape` that is extremely similar to the one actually used in Keras. We give you `__init__`, `__enter__`, and `__exit__`, your job is to implement `GradientTape.gradient`.

:::danger
__Warning:__ This section has historically been difficult for students. It's helpful to carefully consider our hints and the conceptual ideas behind `gradient` __before__ beginning your implementation. You should also freshen up on [Breadth First Search](https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/) to make the implementation easier. 

If you get stuck, feel free to come back to this section later on.
:::

:::info
__Task 1 [GradientTape.gradient]:__ Implement the `gradient` method to compute the gradient of the loss with respect to each of the trainable params. The output of this method should be a list of gradients, one for each of the trainable params.
:::

:::success
__Hint:__ You'll utilize the `compose_input_gradients` and `compose_weight_gradients` methods from `Diffable` in `core.py` to collect the gradients at every step. 

Keep in mind that you can call `id(<tensor_variable>)` to get the object id of a tensor object!
:::

:::warning
__You should know:__ When you finish this section you will have written a highly generalized gradient method that could handle an arbitrary network. This method will also function almost exactly how Keras implements the `gradient` method. This is a very powerful method but it is just one way to implement autograd.
:::

## 9. beras/model.py
In `beras/model.py` we are going to construct a general `Model` abstract class that we will use to define our `SequentialModel`. The `SequentialModel` simply calls all of it's layers in order for the forward pass.

:::warning
__You should know:__ At first it may seem like all neural nets would be `SequentialModel`s but there are some archetectures like __ResNets__ that break the sequential assumption. 
:::

:::info
__Task 1 [Model.weights]:__ Construct a list of all weights in the model and return it.
:::

:::success
We give you `Model.compile`, which just sets the optimizer, loss and accuracy attributes in the model. In Keras, compile is a huge method that prepares these components to make them hyper-efficient. That implementation is highly technical and outside the scope of the course but feel free to look into it if you are interested. 
:::
:::info
__Task 2 [Model.fit]:__ This method should train the model for the number of `epochs` given on the train and test data, `x` and `y` given with batch size, `batch_size`. Importantly, you want make sure you record the metrics throughout training and print stats out during the train so that you can watch the metrics as the model trains.

You can use the `print_stats` and `update_metric_dict` functions provided. Note that neither of these methods return any values, `print_stats` prints out the values directly and `update_metric_dict(super_dict, sub_dict)` updates `super_dict` with the mean metrics from `sub_dict`.

__Note:__ You do __not__ need to call the model here, you should instead use `self.batch_step(...)` which all child classes of `Model` will implement. 
:::

:::info
__Task 3 [Model.evaluate]:__ This method should look _very similar_ to `Model.fit` except we need to ensure the model does not train on the testing data. Additionally, we will test on the entirety of the test set one time, so there is no need for the epochs parameter from `Model.fit`.
:::

:::info
__Task 4 [SequentialModel.forward]:__ This method passes the input through each layer in `self.layers` sequentially. 
:::

:::info
__Task 5 [SequentialModel.batch_step]:__ This method trains makes a model prediction and computes the loss __inside of GradientTape__ just like you did in HW2. Be sure to use the `training` argument to adjust the weights of the model only when `training` is True. This method should return the _loss and accuracy_ for the batch in a dictionary.

**Note**: you should have used the implict `__call__` function for the layers in the forward method to ensure that each layer gets tracked to the GradientTape (i.e. `layer(x)`). However, because we don't define how to calculate gradients for a SequentialModel, make sure to use the SequentialModel's `forward` method in `batch_step`.
:::

## 10. `preprocess.py`
In this section you will fill out the `load_and_preprocess_data()` function that will load in, flatten, normalize and convert all the data into `Tensor`s.

:::info
__Task 1 [load_and_preprocess_data()]:__ We provide the code to load in the data, your job is to 
1. Normalize the values so that they are between 0 and 1
2. Flatten the arrays such that they are of shape (number of examples, 28*28).
3. Convert the arrays to `Tensor`s and return the train inputs, train labels, test inputs and test labels __in that order__.

4. You should NOT shuffle the data in this method or do any other transformations than what we describe in 1-3. Importantly, you should NOT return one_hot labels. You'll create those when training and testing.
:::

## 11. `assignment.py`
Here you put it all together! 

:::info
__Task 1 [Create, Train, and Test your model]:__ You'll find 4 mostly empty functions in `assignment.py`: `get_model`, `get_optimizer`, `get_loss_fn`, and `get_acc_fn`. You should fill these out to create your model however you'd like, you may have to play with different Dense layers, activations, optimizers, etc. to get better accuracies.

Once you have those 4 methods filled out, you can fill out the `__main__` block to train the model. The steps are outlined in the stencil.

You are looking for an accuracy >= 95% within 10 epochs consistently. **Once you are ready**, you can submit your code to the autograder with an additional `FINAL.txt` file which tells the autograder that you'd like to train and test for score on the autograder. **The autograder will use your `get_model`, `get_loss_fn` and `get_optimizer` to initialize your model on gradescope.** This will take some time on gradescope so expect **~10 minutes** waiting time. That said, if you are consistently able to get up to accuracy locally and are passing all other tests you should have no trouble on gradescope.
:::

## Submission

### Requirements
You'll need to submit all the files associated with the assignment, and a "predictions.npy" containing your best model predictions.

:::warning
Please make sure to fill out this [form](https://forms.gle/QmoiEMAJvLwfHrh36) if you used Gen AI to help you on this assignment.
:::

### Grading

Your code will be primarily graded on functionality, as determined by the Gradescope autograder.

:::warning
You will not receive any credit for functions that use `tensorflow`, `keras`, `torch`, or `scikit-learn` functions within them. You must implement all functions manually using either vanilla Python or NumPy.
 :::

### Handing In

You should submit the assignment via Gradescope under the corresponding project assignment through Github or by submitting all files individually. 

To submit via Github, commit and push all of your changes to your repository to GitHub. You can do this by running the following commands.

```bash
git commit -am "commit message"
git push
```

For those of y'all who are already familiar with `git`: the `-am` flag to `git commit` is a pretty cool shortcut which adds all of our modified files and commits them with a commit message.

**Note:** We highly recommend committing your files to git and syncing with your GitHub repository **often** throughout the course of the assignment to ensure none of your hard work is **lost**!

### Leaderboard

There is a leaderboard active for this assignment, __this is purely for fun and is not graded.__