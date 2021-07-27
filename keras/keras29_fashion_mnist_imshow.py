from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

for i in range(10):
    print(x_train[i])
    print(f'y{[i]} 값 : ', y_train[i])
    plt.imshow(x_train[i], 'gray')
    plt.show()
