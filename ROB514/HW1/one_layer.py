import numpy as np
import csv
import yaml

# sigmoid functions
def sigmoid(x):
    return( 1/ (1 + np.exp(-x) )  )

def sigmoid_prime(x):
    on = np.matrix(np.ones(x.shape))
    return( np.multiply(sigmoid(x), on - sigmoid(x) ) )

# no function
def identity(x):
    return(x)

def identity_prime(iny):
    x = iny.copy()
    with np.nditer(x, op_flags=['readwrite']) as it:
        for n in it:
            n[...] = 1.0

    return(x)
    


# ReLU
def ReLU(iny):
    x = iny.copy()
    with np.nditer(x, op_flags=['readwrite']) as it:
        for n in it:
            n[...] = max(0, n)

    return(x)

def ReLU_prime(iny):
    x = iny.copy()
    with np.nditer(x, op_flags=['readwrite']) as it:
        for n in it:
            if n > 0:
                n[...] = 1
            else:
                n[...] = 0
    return x



class nn_1layer():
    def __init__(self, layer_size_ = 10, activation_ = identity, dact_ = identity, lr_=0.001, bias_ = True):

        self.bias = bias_
        in_size_ = 5
        out_size_ = 2
        
        self.w1 = np.matrix(np.random.rand(layer_size_, in_size_))
        self.b1 = np.matrix(np.random.rand(layer_size_, 1))

        self.w2 = np.matrix(np.random.rand(out_size_, layer_size_))
        self.b2 = np.matrix(np.random.rand(out_size_, 1))

        if(not self.bias):
            self.b1 *= 0
            self.b2 *= 0

        
        self.activation = activation_
        self.act_prime = dact_

        self.lr = lr_

    def feedforward(self, x ):
        
        self.w1 * x
        self.z1 = np.add(self.w1 * x, self.b1)
        self.a1 = self.activation(self.z1)

        self.z2 = np.add(self.w2 * self.a1, self.b2)
        self.a2 = self.activation(self.z2)
        #self.a2/=sum(self.a2)
        
        return( self.a2 )

    def backprop(self,x, y):
        """ Assumed to be ran after feedforward for a given x """


        error = (0.5 * (y - self.a2).transpose() * (y - self.a2) ).item(0)
        

        # deltas
        d2 = np.multiply( (self.a2 - y), self.act_prime(self.z2) )
        d1 = np.multiply( self.w2.transpose() * d2, self.act_prime(self.z1) )
        
        # calculate error
        ew1 = d1 * x.transpose()
        eb1 = d1
        ew2 = d2 * self.a1.transpose()
        eb2 = d2
        

        # update weights
        self.w1 = np.add(self.w1, ew1 * self.lr)
        #return( np.sum( np.pow( np.subtract(self.a2, y), 2) ) )
        self.w2 = np.add(self.w2, -ew2 * self.lr)

        if(self.bias):
            self.b1 = np.add(self.b1, -eb1 * self.lr)
            self.b2 = np.add(self.b2, -eb2 * self.lr)

        # update lr?
        return(error)

class Trainer():
    def __init__(self, trainfile_, testfile_, shuffle_ = True):
        self.shuffle = shuffle_
        with open(trainfile_, 'r') as f:
            reader = csv.reader(f)
            self.train = np.array(list(reader),float)

        with open(testfile_, 'r') as f:
            reader = csv.reader(f)
            self.test = np.array(list(reader), float)
            
    def epoch(self, nn):
        correct_class = 0.0
        mse_error = 0.0
        tot = 0.0
        if(self.shuffle):
            np.random.shuffle(self.train)
        for row in self.train:
            x = np.matrix(row[:5]).transpose()
            y = np.matrix(row[5:]).transpose()
            
            a2 = nn.feedforward(x)
            mse_error += nn.backprop(x,y)
            if( np.argmax(a2) == np.argmax(y) ):
                #print(y, a2, tot)
                correct_class += 1.0
            tot += 1.0

        return(correct_class/tot)

    def testy(self, nn):
        correct_class= 0.0
        tot = 0.0
        for row in self.test:
            x = np.matrix(row[:5]).transpose()
            y = np.matrix(row[5:]).transpose()

            a2 = nn.feedforward(x)
            if( np.argmax(a2) == np.argmax(y) ):
                #print(y, a2, tot)
                correct_class += 1.0
            tot += 1

        return( correct_class/tot )
        

# # iterations
# learning rate
# hidden layer size
# bias or no bias
# activation function


if __name__ == "__main__":
    layer_sizes = [x+2 for x in range(14)]
    activations = [ [identity, identity, 0], [sigmoid, sigmoid_prime, 1],[ReLU, ReLU_prime, 2]]
    lrs = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    bias = [True, False]
    shuffle = [True, False]

    epochs = 1000


    nns = []
    for ls in layer_sizes:
        for acts in activations:
            for lr in lrs:
                for bia in bias:
                    for shu in shuffle:
                        nns.append( [nn_1layer(ls, acts[0], acts[1], lr, bia), [shu, ls, acts[2], lr, bia], [] ] )

    dataSet = "1"
    for dataSet in ["1", "2"]:
        print(dataSet)
        rand_trainer  =  Trainer("hw1/train" + dataSet + ".csv", "hw1/test" + dataSet + ".csv", True)
        same_trainer  =  Trainer("hw1/train" + dataSet + ".csv", "hw1/test" + dataSet + ".csv", False)
        
        for i in range(epochs):
            print("Epoch:", i )
            for nn in nns:
                #print(nn[1])
                if( nn[1][0] ): # if shuffle
                    train_error = rand_trainer.epoch(nn[0])
                    test_error = rand_trainer.testy(nn[0])
                else:
                    train_error = same_trainer.epoch(nn[0])
                    test_error = same_trainer.testy(nn[0])
                
                    nn[2].append([train_error, test_error])
                    #if( i == epochs - 1):
                    #    print(nn[1], train_error, test_error)

            output = {}
    
            for i in range(len(nns)):
                data = {'meta': nns[i][1], 'output':nns[i][2]}
                output[i] = data
                
            with open('raw_yaml' + dataSet + '.yaml', 'w') as fi:
                yaml.dump(output, fi)
        
        
                                            
