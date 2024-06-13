import gzip
import numpy as np
import os
import struct

# 加载训练和验证数据集
def load_train_images():
    with gzip.open('dataset/train-images-idx3-ubyte.gz', 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    
def load_train_labels():
    with gzip.open('dataset/train-labels-idx1-ubyte.gz', 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8)

def load_verify_images():
    with gzip.open('dataset/t10k-images-idx3-ubyte.gz', 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    
def load_verify_labels():
    with gzip.open('dataset/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8)

# 加载数据集
train_images_defalt = load_train_images()
train_labels_defalt = load_train_labels()
verify_images_defalt = load_verify_images()
verify_labels_defalt = load_verify_labels()

# One-hot编码
train_labels_one_hot = np.zeros((train_labels_defalt.size, train_labels_defalt.max() + 1))
train_labels_one_hot[np.arange(train_labels_defalt.size), train_labels_defalt] = 1

verify_labels_one_hot = np.zeros((verify_labels_defalt.size, verify_labels_defalt.max() + 1)) 
verify_labels_one_hot[np.arange(verify_labels_defalt.size), verify_labels_defalt] = 1 

# 数据预处理
train_images_flatten = train_images_defalt.reshape(train_images_defalt.shape[0], -1)
verify_images_flatten = verify_images_defalt.reshape(verify_images_defalt.shape[0], -1)

train_images_normalized = train_images_flatten / 255
verify_images_normalized = verify_images_flatten / 255

train_images = train_images_normalized
verify_images = verify_images_normalized
train_labels = train_labels_one_hot
verify_labels = verify_labels_one_hot

# 定义激活函数和损失函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

# 定义导数函数
def sigmoid_derivative(x):
    return x * (1 - x)

def softmax_and_cross_entropy_derivative(y, y_hat):
    return y_hat - y

def matrix_derivative(x, delta):
    return np.dot(x.T, delta)

# 初始化参数
input_size = 28 * 28
hidden_size = 32
output_size = 10
learning_rate = 1e-3

np.random.seed(0)

# He初始化 - 适合ReLU激活函数 (比下边的效果好)
weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / output_size)

# Xavier初始化 - 适合sigmoid和tanh激活函数
# weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / (input_size + hidden_size))
# weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / (hidden_size + output_size))

# 权重的初始化值不能都一样, 不然每个感知器的更新都一样, 抓取的特征也一样, 没办法学习

bias_input_hidden = np.zeros(hidden_size) 
bias_hidden_output = np.zeros(output_size) 

# 前向传播
def forward(x):
    global weights_input_hidden, weights_hidden_output, bias_input_hidden, bias_hidden_output
    
    hidden_layer_input = np.dot(x, weights_input_hidden) + bias_input_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_hidden_output
    output_layer_output = softmax(output_layer_input)
    
    return hidden_layer_input, hidden_layer_output, output_layer_input, output_layer_output

# 训练函数
def train(x, y):
    global weights_input_hidden, weights_hidden_output, bias_input_hidden, bias_hidden_output, learning_rate
    
    hidden_layer_input, hidden_layer_output, output_layer_input, output_layer_output = forward(x)
    
    loss = cross_entropy_loss(y, output_layer_output)
    
    delta_hidden_output = softmax_and_cross_entropy_derivative(y, output_layer_output)
    weights_hidden_output_gradient = matrix_derivative(hidden_layer_output, delta_hidden_output)
    weights_hidden_output -= learning_rate * weights_hidden_output_gradient
    bias_hidden_output_gradient = softmax_and_cross_entropy_derivative(y, output_layer_output)
    bias_hidden_output -= learning_rate * bias_hidden_output_gradient.mean(axis=0)
    
    delta_input_hidden = np.dot(delta_hidden_output, weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)
    weights_input_hidden_gradient = matrix_derivative(x, delta_input_hidden)
    weights_input_hidden -= learning_rate * weights_input_hidden_gradient
    bias_input_hidden_gradient = delta_input_hidden.sum(axis=0)
    bias_input_hidden -= learning_rate * bias_input_hidden_gradient
    
    return loss

# 验证函数
def verify(pred_label, verify_label):
    pred_label = np.argmax(pred_label, axis=1)
    verify_label = np.argmax(verify_label, axis=1)
    accuracy = np.sum(pred_label == verify_label) / verify_label.size
    return accuracy

# mini-batch训练
batch_size = 60

for epoch in range(200):
    shuffle_index = np.random.permutation(60000)
    train_images = train_images[shuffle_index]
    train_labels = train_labels[shuffle_index]
    
    for index in range(0, 60000, batch_size):
        x_batch = train_images[index:index + batch_size]
        y_batch = train_labels[index:index + batch_size]
        loss = train(x_batch, y_batch)
        
        if index % 10000 == 0:
            print(f'epoch: {epoch}, index: {index}, loss: {loss}')
            print(f'accuracy: {verify(forward(verify_images)[3], verify_labels)}')
