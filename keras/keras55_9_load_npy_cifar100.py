import numpy as np  


x_train_data = np.load('./_save/_npy/k55_x_train_data_cifar100.npy')
x_test_data = np.load('./_save/_npy/k55_x_test_data_cifar100.npy')
y_train_data = np.load('./_save/_npy/k55_y_train_data_cifar100.npy')
y_test_data = np.load('./_save/_npy/k55_y_test_data_cifar100.npy')

print(x_train_data.shape, y_train_data.shape) # (50000, 32, 32, 3) (50000, 1) 


