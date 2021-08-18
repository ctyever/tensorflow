import numpy as np 

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100, 5000],
                [1000,2000,3,4000,5000,6000,7000,8,9000,10000, 1001]])

# (2, 10) -> (10, 2)

aaa = aaa.transpose()
print(aaa.shape)

def outliers(data_out):
    quartile_1, q2, quertile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quertile_3)
    iqr = quertile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quertile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(aaa)

print('이상치의 위치 : ', outliers_loc)

import matplotlib.pyplot as plt

plt.boxplot(aaa[:, 0], sym="bo")
# plt.boxplot(aaa[:, 1], sym="bo")
plt.show()