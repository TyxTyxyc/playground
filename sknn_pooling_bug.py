from sklearn import datasets, cross_validation
from sknn.mlp import Classifier, Layer, Convolution

digits = datasets.load_digits()
X = digits.images
y = digits.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

conv_layer_1 = Convolution(
    type='Rectifier',
    name='Conv_Layer_1',
    kernel_shape=(3, 3),
    channels=8,
    border_mode='full',
    pool_shape=(2, 2)
    )

dense_layer_1 = Layer(
    type='Rectifier',
    name='Dense_Layer_1',
    units=128,
    )

output_layer = Layer(
    type='Softmax',
    name='Output_Layer',
    )

net = Classifier(
    layers=[conv_layer_1,
            dense_layer_1,
            output_layer],
    learning_rate=0.003,
    n_iter=10,
    n_stable=10,
    verbose=True,
    learning_rule='momentum',
    batch_size=10,
    valid_size=0.1,
    weight_decay=0.0005,
    dropout_rate=0.03
    )

net.fit(X_train, y_train)
