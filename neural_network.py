import numpy as np
import scipy.special


class neural_network:

    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):

        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learning_rate

        self.wih = np.random.randn(self.hnodes, self.inodes)
        self.who = np.random.randn(self.onodes, self.hnodes)

        self.activation_function = lambda x: scipy.special.expit(x)

    def query(self,input):
        inputs = np.array(input,ndmin=2).T

        hidden_inputs = np.dot(self.wih,inputs)
        hidden_output = self.activation_function(hidden_inputs)

        final_input = np.dot(self.who,hidden_output)
        final_output = self.activation_function(final_input)

        return final_output

    def train(self,input,target):
        inputs = np.array(input,ndmin=2).T
        targets = np.array(target,ndmin=2).T

        hidden_inputs = np.dot(self.wih,inputs)
        hidden_output = self.activation_function(hidden_inputs)

        final_input = np.dot(self.who,hidden_output)
        final_output = self.activation_function(final_input)

        error = targets-final_output

        hidden_error = np.dot(self.who.T,error)

        self.who += self.lr*(np.dot(error*final_output*(1-final_output),np.transpose(hidden_output)))
        self.wih += self.lr*(np.dot(hidden_error*hidden_output*(1-hidden_output),np.transpose(inputs)))

    def rev_query(self,input):
        inputs = np.array(input,ndmin=2).T

        layer3_out = np.dot(self.who,inputs)
        layer2_in = self.activation_function(layer3_out)

        layer1_in = np.dot(self.wih,layer2_in)
        layer1_out = self.activation_function(layer1_in)

        return(layer1_out)