'''
There are two ways to define a dense layer in tensorflow. 
The first involves the use of low-level, linear algebraic operations. 
The second makes use of high-level keras operations. 
In this exercise, we will use the first method to construct the network shown in the image below.

The input layer contains 3 features -- education, marital status, and age -- which are available as borrower_features. 
The hidden layer contains 2 nodes and the output layer contains a single node.

For each layer, you will take the previous layer as an input, initialize a set of weights, compute the product of the inputs and weights, and then apply an activation function. 
Note that Variable(), ones(), matmul(), and keras() have been imported from tensorflow.
'''

# Initialize bias1
bias1 = Variable(1.0)

# Initialize weights1 as 3x2 variable of ones
weights1 = Variable(ones((3, 2)))

# Perform matrix multiplication of borrower_features and weights1
product1 = matmul(borrower_features, weights1)

# Apply sigmoid activation function to product1 + bias1
dense1 = keras.activations.sigmoid(product1 + bias1)

# Print shape of dense1
print("\n dense1's output shape: {}".format(dense1.shape))

# From previous step
bias1 = Variable(1.0)
weights1 = Variable(ones((3, 2)))
product1 = matmul(borrower_features, weights1)
dense1 = keras.activations.sigmoid(product1 + bias1)

# Initialize bias2 and weights2
bias2 = Variable(1.0)
weights2 = Variable(ones((2, 1)))

# Perform matrix multiplication of dense1 and weights2
product2 = matmul(dense1, weights2)

# Apply activation to product2 + bias2 and print the prediction
prediction = keras.activations.sigmoid(product2 + bias2)
print('\n prediction: {}'.format(prediction.numpy()[0,0]))
print('\n actual: 1')


'''
In this exercise, we'll build further intuition for the low-level approach by constructing the first dense hidden layer for the case where we have multiple examples. 
We'll assume the model is trained and the first layer weights, weights1, and bias, bias1, are available. 
We'll then perform matrix multiplication of the borrower_features tensor by the weights1 variable. 
Recall that the borrower_features tensor includes education, marital status, and age. 
Finally, we'll apply the sigmoid function to the elements of products1 + bias1, yielding dense1.
'''

# Compute the product of borrower_features and weights1
products1 = matmul(borrower_features, weights1)

# Apply a sigmoid activation function to products1 + bias1
dense1 = keras.activations.sigmoid(products1 + bias1)

# Print the shapes of borrower_features, weights1, bias1, and dense1
print('\n shape of borrower_features: ', borrower_features.shape)
print('\n shape of weights1: ', weights1.shape)
print('\n shape of bias1: ', bias1.shape)
print('\n shape of dense1: ', dense1.shape)


'''
We've now seen how to define dense layers in tensorflow using linear algebra. 
In this exercise, we'll skip the linear algebra and let keras work out the details. 
This will allow us to construct the network below, which has 2 hidden layers and 10 features, using less code than we needed for the network with 1 hidden layer and 3 features.
'''

# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)

# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)


'''
In this exercise, you will again make use of credit card data. 
The target variable, default, indicates whether a credit card holder defaults on his or her payment in the following period. 
Since there are only two options--default or not--this is a binary classification problem. 
While the dataset has many features, you will focus on just three: the size of the three latest credit card bills. 
Finally, you will compute predictions from your untrained network, outputs, and compare those the target variable, default.

The tensor of features has been loaded and is available as bill_amounts. 
Additionally, the constant(), float32, and keras.layers.Dense() operations are available.
'''

# Construct input layer from features
inputs = constant(bill_amounts, float32)

# Define first dense layer
dense1 = keras.layers.Dense(3, activation='relu')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(2, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print error for first five examples
error = default[:5] - outputs.numpy()[:5]
print(error)


'''
In this exercise, we expand beyond binary classification to cover multiclass problems. 
A multiclass problem has targets that can take on three or more values. 
In the credit card dataset, the education variable can take on 6 different values, each corresponding to a different level of education. 
We will use that as our target in this exercise and will also expand the feature set from 3 to 10 columns.

As in the previous problem, you will define an input layer, dense layers, and an output layer. 
You will also print the untrained model's predictions, which are probabilities assigned to the classes. 
The tensor of features has been loaded and is available as borrower_features. 
Additionally, the constant(), float32, and keras.layers.Dense() operations are available.
'''

# Construct input layer from borrower features
inputs = constant(borrower_features, float32)

# Define first dense layer
dense1 = keras.layers.Dense(10, activation='sigmoid')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(8, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(6, activation='softmax')(dense2)

# Print first five predictions
print(outputs.numpy()[:5])


'''
Consider the plot of the following loss function, loss_function(), which contains a global minimum, marked by the dot on the right, and several local minima, including the one marked by the dot on the left.

In this exercise, you will try to find the global minimum of loss_function() using keras.optimizers.SGD(). 
You will do this twice, each time with a different initial value of the input to loss_function(). 
First, you will use x_1, which is a variable with an initial value of 6.0. 
Second, you will use x_2, which is a variable with an initial value of 0.3. 
Note that loss_function() has been defined and is available.
'''

# Initialize x_1 and x_2
x_1 = Variable(6.0,float32)
x_2 = Variable(0.3,float32)

# Define the optimization operation
opt = keras.optimizers.SGD(learning_rate=0.01)

for j in range(100):
	# Perform minimization using the loss function and x_1
	opt.minimize(lambda: loss_function(x_1), var_list=[x_1])
	# Perform minimization using the loss function and x_2
	opt.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())


'''
The previous problem showed how easy it is to get stuck in local minima. 
We had a simple optimization problem in one variable and gradient descent still failed to deliver the global minimum when we had to travel through local minima first. 
One way to avoid this problem is to use momentum, which allows the optimizer to break through local minima. 
We will again use the loss function from the previous problem, which has been defined and is available for you as loss_function().

Several optimizers in tensorflow have a momentum parameter, including SGD and RMSprop. 
You will make use of RMSprop in this exercise. 
Note that x_1 and x_2 have been initialized to the same value this time. 
Furthermore, keras.optimizers.RMSprop() has also been imported for you from tensorflow.
'''

# Initialize x_1 and x_2
x_1 = Variable(0.05,float32)
x_2 = Variable(0.05,float32)

# Define the optimization operation for opt_1 and opt_2
opt_1 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.00)

for j in range(100):
	opt_1.minimize(lambda: loss_function(x_1), var_list=[x_1])
    # Define the minimization operation for opt_2
	opt_2.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())


'''
A good initialization can reduce the amount of time needed to find the global minimum. 
In this exercise, we will initialize weights and biases for a neural network that will be used to predict credit card default decisions. 
To build intuition, we will use the low-level, linear algebraic approach, rather than making use of convenience functions and high-level keras operations. 
We will also expand the set of input features from 3 to 23. 
Several operations have been imported from tensorflow: Variable(), random(), and ones().
'''

# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]))

# Initialize the layer 1 bias
b1 = Variable(ones([7]))

# Define the layer 2 weights
w2 = Variable(random.normal([7, 1]))

# Define the layer 2 bias
b2 = Variable(ones([0]))


'''
In this exercise, you will train a neural network to predict whether a credit card holder will default. 
The features and targets you will use to train your network are available in the Python shell as borrower_features and default. 
You defined the weights and biases in the previous exercise.

Note that the predictions layer is defined as 
σ(layer1 * w2 + b2), where σ is the sigmoid activation, layer1 is a tensor of nodes for the first hidden dense layer, w2 is a tensor of weights, and b2 is the bias tensor.

The trainable variables are w1, b1, w2, and b2. 
Additionally, the following operations have been imported for you: keras.activations.relu() and keras.layers.Dropout().
'''

# Define the model
def model(w1, b1, w2, b2, features = borrower_features):
	# Apply relu activation functions to layer 1
	layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout rate of 0.25
	dropout = keras.layers.Dropout(0.25)(layer1)
	return keras.activations.sigmoid(matmul(dropout, w2) + b2)

# Define the loss function
def loss_function(w1, b1, w2, b2, features = borrower_features, targets = default):
	predictions = model(w1, b1, w2, b2)
	# Pass targets and predictions to the cross entropy loss
	return keras.losses.binary_crossentropy(targets, predictions)


'''
In the previous exercise, you defined a model, model(w1, b1, w2, b2, features), and a loss function, loss_function(w1, b1, w2, b2, features, targets), both of which are available to you in this exercise. 
You will now train the model and then evaluate its performance by predicting default outcomes in a test set, which consists of test_features and test_targets and is available to you. 
The trainable variables are w1, b1, w2, and b2. 
Additionally, the following operations have been imported for you: keras.activations.relu() and keras.layers.Dropout().
'''

# Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2), 
                 var_list=[w1, b1, w2, b2])

# Make predictions with model using test features
model_predictions = model(w1, b1, w2, b2, test_features)

# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)


'''
In chapter 3, we used components of the keras API in tensorflow to define a neural network, but we stopped short of using its full capabilities to streamline model definition and training. 
In this exercise, you will use the keras sequential model API to define a neural network that can be used to classify images of sign language letters. 
You will also use the .summary() method to print the model's architecture, including the shape and number of parameters associated with each layer.

Note that the images were reshaped from (28, 28) to (784,), so that they could be used as inputs to a dense layer. 
Additionally, note that keras has been imported from tensorflow for you.
'''

# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Print the model architecture
print(model.summary())


