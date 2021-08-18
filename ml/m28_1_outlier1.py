# 이상치 처리
# 1. 삭제
# 2. Nan 처리후 -> 보간 / linear
# 3.  .................. (결측시 처리 방법과 유사)
# 4. scaler -> Robustscaler, QuantileTransformer ... 등등
# 5. 모델링 : tree 계열... DT, RF, XG, LGBM...

import numpy as np 
aaa = np.array([1,2,-1000, 4,5,6,7,8,90,100,500])

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



# 시각화
# 실습
# 위 데이터를 boxplot 으로 그리시오!
import matplotlib.pyplot as plt

plt.boxplot(aaa, sym="bo")
plt.show()