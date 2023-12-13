# Image-Classification
This helps in the classification of images of 2 types i.e happy and sad

## Some descriptions about code used in the project
### tf.keras.utils.image_dataset_from_directory()
`tf.keras.utils.image_dataset_from_directory` is a generator-like function in TensorFlow, and it doesn't actually load all the data into memory at once. Instead, it creates a tf.data.Dataset, which is a TensorFlow data pipeline that efficiently loads and processes data in a streaming fashion, typically on-the-fly during training.

Instead of loading everything in the memory, it allows us to create a data pipeline which allows us to scale out to much larger datasets but it also gives us a repeatable set of steps.  

This step sets up a pipeline for efficiently loading and processing image data during training.

### as_numpy_iterator()
Returns an iterator which converts all elements of the dataset to numpy. Use as_numpy_iterator to inspect the content of your dataset. To see element shapes and types, print dataset elements directly instead of using `as_numpy_iterator`.

The as_numpy_iterator() method is used to convert the TensorFlow dataset (data) into a NumPy iterator (data_iterator). This is helpful when you want to iterate through the dataset and obtain NumPy arrays for further processing or analysis. It's a convenient way to work with the data in a NumPy-friendly format.

The conversion of a TensorFlow dataset to a NumPy iterator using `as_numpy_iterator()` is a convenience step that allows you to work with the data in a NumPy-friendly format during certain operations.

### next()
The `next()` function in Python is used to fetch the next item from an iterator. In the context of TensorFlow datasets, calling `next()` on the iterator is a way to retrieve the next batch of data. Let's understand why it's useful:

1. Iteration through Batches: When you have a large dataset and you want to train a machine learning model using stochastic gradient descent (SGD) or mini-batch gradient descent, you need to iterate through the dataset in batches. The `next()` function allows you to get the next batch in each iteration of your training loop.

2. Streaming Data: TensorFlow datasets are designed to efficiently handle large datasets that might not fit into memory. Instead of loading the entire dataset at once, the `next()` function allows you to fetch batches of data on-the-fly during training. This is particularly important when working with large datasets that cannot be loaded entirely into memory.

3. Sequential Processing: The `next()` function is typically used in a loop, where each iteration corresponds to processing a different batch. This sequential processing is essential for updating model parameters and training deep learning models.

In this example, after converting the dataset to a NumPy iterator, `data_iterator.next()` is used to fetch the next batch of data. This batch is then used for training or other processing.
 
In summary, using `next()` facilitates the sequential processing of batches during training, allowing you to efficiently train machine learning models on large datasets in a memory-friendly manner.

### tf.keras.callbacks.Tensorboard()
A callback is an object that can perform actions at various stages of training (e.g. at the start or end of an epoch, before or after a single batch, etc).

Callbacks in TensorFlow Keras can be used to stop the training of the model when a certain training accuracy is achieved, or when the loss goes below a certain value. They can also be used to save models after every epoch or to create customized saving checkpoints. A callback function can be called
- at the beginning or end of every epoch
- at the beginning or end of the training
- at the beginning and end of the training batch

### hist 
Here we are specifying that we want to log out all our information at tensor board and storing it inside a variable called hist or history. We can use that to plot the information into a graph. 

## Some important websites to refer from
https://neptune.ai/blog/tensorboard-tutorial

https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
