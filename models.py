#Shreya Shukla (ss4515) and Medhasri Veldurthi (mv670)
from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module

"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        "*** YOUR CODE HERE ***"
        self.w = Parameter(ones(1,dimensions)) #Initialize your weights here
        #self.b = 0 #optional: initialize bias to 0 and change if needed

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        #score = w·x + b
        return tensordot(self.w, x.T, dims=1) #+ self.b #optional


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if self.run (x) >= 0:
            return 1
        else:
            return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"
            converged = False
            while not converged:
                #assume the values have converged, which will break the loop at next iteration unless its changed again
                converged = True   
                for batch in dataloader:
                    x = batch['x']
                    label = batch['label']

                    prediction = self.get_prediction(x)
                    if (prediction != label):
                        #update the weights
                        self.w += label * x
                        converged = False #continue the loop becauuse the values have not converged


class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super().__init__()

        #setting up the single hidden layer and the output
        self.hidden_layer = Linear(1, 100)
        self.output = Linear(100, 1)

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        #hidden layer activation
        hidden_layer_activate = self.hidden_layer(x)
        #activate with the relu
        activated = relu(hidden_layer_activate)
        #activating the output layer for the final node
        node = self.output(activated) 
        return node
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
        
        #getting the predicted y
        predicted_y = self(x)
        #then calculating the loss
        loss = mse_loss(predicted_y, y)
        return loss
 
  

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        "*** YOUR CODE HERE ***"

        #the learning rate will be 0.01, batch size is 20 which is evenly divisible
        learning_rate=0.01
        dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

        #running through dataset for training 1500 times
        for i in range(1500): 
            #variable to keep tracsk of the total loss
            total_loss = 0.0
            for batch in dataloader:

                #getting the features and the label
                x = batch['x']
                y = batch['label']

                self.zero_grad()

                #getting the loss and keeping track of the totalloss
                loss = self.get_loss(x, y)
                total_loss += loss.item() 

                #calculating the gradients
                loss.backward()

                #updating based on gradient if it exists
                for param in self.parameters():
                    if param.grad is not None:
                        param.data = param.data - learning_rate * param.grad.data



class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        #define the different layers of the neural network
        self.layer1 = Linear(input_size, 128)
        self.layer2 = Linear(128, 64)
        self.layer3 = Linear(64, output_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        activate_L1 = relu(self.layer1(x))
        activate_L2 = relu(self.layer2(activate_L1))
        outputs = self.layer3(activate_L2)

        return outputs

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        v = self.run(x)
        loss = cross_entropy(v, y)
        return loss

        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """

        learning_rate = 0.005
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        num_epochs = 10

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
            #getting the features and the label
                x = batch['x']
                y = batch['label']

                #reset the gradients calculated by pytorch
                optimizer.zero_grad()

                #getting the loss and keeping track of the totalloss
                loss = self.get_loss(x, y)
                total_loss += loss.item() 

                #backpropagation
                loss.backward()         

                #update your weights 
                optimizer.step()
