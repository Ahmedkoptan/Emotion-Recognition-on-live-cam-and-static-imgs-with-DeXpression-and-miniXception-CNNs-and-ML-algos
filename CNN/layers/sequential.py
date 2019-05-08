from __future__ import print_function
import numpy as np
import cPickle
import traceback


class Sequential(object):
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        # set name from variable name. http://stackoverflow.com/questions/1690400/getting-an-instance-name-inside-class-init
        (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
        def_name = text[:text.find('=')].strip()
        self.name = def_name

        try:
            self.load()
        except:
            ##############
            # to demonstrate
            self.someAttribute = 'bla'
            self.someAttribute2 = ['more']
            ##############

            self.save()


    def save(self):
        """save class as self.name.txt"""
        file = open(self.name+'.txt','w')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def load(self):
        """try load self.name.txt"""
        file = open(self.name+'.txt','r')
        dataPickle = file.read()
        file.close()

        self.__dict__ = cPickle.loads(dataPickle)
    def forward(self, x, target=None):
        temp = x.copy()
        for i in range(len(self.layers)):
            u = self.layers[i].forward(temp)
            temp = u.copy()
        output=u

        if target is None:
            output
        else:
            output=self.loss.forward(u, target)


        return output

    def backward(self): ####
        grad = self.loss.backward()
        for i in range(len(self.layers)):
            grad = self.layers[len(self.layers)-1-i].backward(grad)

        return grad

    def update_param(self, lr): ####
        for i in range(len(self.layers)):
            self.layers[i].update_param(lr)


    def fit(self, x, y, epochs=10, lr=0.1, batch_size=128):
        batch_numbers=int(len(x[:,0])/batch_size)
        losss=np.zeros(epochs)
        for i in range(epochs):
            print("Starting epoch "+str(i))
            l=np.zeros(batch_numbers)
            for j in range(batch_numbers):
                l[j] = self.forward(x[j*batch_size:(j+1)*batch_size,:],y[j*batch_size:(j+1)*batch_size,:])
                self.backward()
                self.update_param(lr)
            losss[i] = np.mean(l)
            print("Epoch "+str(i)+" is done")
        return losss

    def predict(self, x):
        p=self.forward(x)
        pred=np.argmax(p,axis=1)
        prediction = np.zeros([x.shape[0],7])
        for i in range(x.shape[0]):
            prediction[i, pred[i]] = 1
        return prediction

