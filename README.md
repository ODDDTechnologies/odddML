# odddML

A first iteration of a deep learning library that contains custom layers and Model templates implemented with the subclass API of Tensorflow 2.0. We are populating the library with some basic implementations and we aim to expand it into a full distribution with a ton of our implementations by the end of 2020.

### Instalation

```bash
pip install odddml
```
Will install odddml and all of the dependencies.


### Machine Learning with odddML

```python
from odddML.templateModels import SimpleConv2DClassifier
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # load a sandbox dataset from tensorflow

x_train = x_train / 255 #normalize 
x_test= x_test / 255

model = SimpleConv2DClassifier(num_classes=len(np.unique(y_train))) # initialize the model with the number of classes of the problem

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #set the optimizer, loss and metrics
model.fit(x_train, y_train, validation_data=(x_test, y_test))
# done.. 
```

If you want to contribute feel free to ping me. nick@odddtech.com
