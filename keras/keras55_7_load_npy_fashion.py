import numpy as np  


x_train_data = np.load('./_save/_npy/k55_x_train_data_fashion.npy')
x_test_data = np.load('./_save/_npy/k55_x_test_data_fashion.npy')
y_train_data = np.load('./_save/_npy/k55_y_train_data_fashion.npy')
y_test_data = np.load('./_save/_npy/k55_y_test_data_fashion.npy')

print(x_train_data.shape, y_train_data.shape) # (60000, 28, 28) (60000,) 


