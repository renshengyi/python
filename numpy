numpy库
一、简介
NumPy 是一个 Python 包。 它代表 “Numeric Python”。 它是一个由多维数组对象和用于处理数组的例程集合组成的库。

二、NumPy 操作
使用NumPy，开发人员可以执行以下操作：
数组的算数和逻辑运算。
傅立叶变换和用于图形操作的例程。
与线性代数有关的操作。 NumPy 拥有线性代数和随机数生成的内置函数。

三、
1.创建矩阵（采用ndarray对象） np.arrary([])
import numpy as np #引入numpy库
#创建一维的narray对象
a = np.array([1,2,3,4,5])
#创建二维的narray对象
a2 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
#创建多维对象以其类推 

2.获取矩阵行数列数（二维情况） a.shape
import numpy as np
a = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(a.shape) #结果返回一个tuple元组 (2L, 5L)
print(a.shape[0]) #获得行数，返回 2
print(a.shape[1]) #获得列数，返回 5

3.矩阵的截取
1)按行列截取  a[line1:line2,column1:column2]
#矩阵的截取和list相同，可以通过[]（方括号）来截取
import numpy as np
a = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(a[0:1]) #截取第一行,返回 [[1 2 3 4 5]]
print(a[1,2:5]) #截取第二行，第三、四列，返回 [8 9]
print(a[1,:]) #截取第二行,返回 [ 6  7  8  9 10]
2)按条件截取 a[condition]
#按条件截取其实是在[]（方括号）中传入自身的布尔(bool)语句 
import numpy as np
a = np.array([[1,2,3,4,5],[6,7,8,9,10]])
b = a[a>6] # 截取矩阵a中大于6的元素，范围是一维数组
print(b) # 返回 [7  8  9 10]
# 其实布尔语句首先生成一个布尔矩阵，将布尔矩阵传入[]（方括号）实现截取
print(a>6) 
# 返回
[[False False False False False]
 [False  True  True  True  True]]
按条件截取应用较多的是对矩阵中满足一定条件的元素变成特定的值。 
例如将矩阵中大于6的元素变成0。
import numpy as np
a = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(a)
#开始矩阵为
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]]
a[a>6] = 0
print(a)
#大于6清零后矩阵为
[[1 2 3 4 5]
 [6 0 0 0 0]]

3.矩阵的合并 np.hstack([a1,a2]) or np.vstack((a1,a2))
#矩阵的合并可以通过numpy中的hstack方法和vstack方法实现
import numpy as np
a1 = np.array([[1,2],[3,4]])
a2 = np.array([[5,6],[7,8]])
#!注意 参数传入时要以列表list或元组tuple的形式传入
print(np.hstack([a1,a2])) 
#横向合并，返回结果如下 
[[1 2 5 6]
 [3 4 7 8]]
print(np.vstack((a1,a2)))
#纵向合并，返回结果如下
[[1 2]
 [3 4]
 [5 6]
 [7 8]]

4.通过函数创建矩阵
numpy模块中自带了一些创建ndarray对象的函数，可以很方便的创建常用的或有规律的矩阵。
1)arange  np.arrange(a,b,space)
import numpy as np
a = np.arange(10) # 默认从0开始到10（不包括10），步长为1
print(a) # 返回 [0 1 2 3 4 5 6 7 8 9]
a1 = np.arange(5,10) # 从5开始到10（不包括10），步长为1
print(a1) # 返回 [5 6 7 8 9]
a2 = np.arange(5,20,2) # 从5开始到20（不包括20），步长为2
print(a2) # 返回 [ 5  7  9 11 13 15 17 19]

2)linspace 等差数列 np.linspace(a,b,space)
import numpy as np
a = np.linspace(0,10,7) # 生成首位是0，末位是10，含7个数的等差数列
print(a) 
# 结果 
[  0.           1.66666667   3.33333333   5.         6.66666667  8.33333333  10.        ]

3)logspace 等比数列 np.logspace(a,b,number)   含有number个数10^a-10^b的等比数列
import numpy as np
a = np.logspace(0,2,5)#生成首位是10^0，末位是10^2，含5个数的等比数列
print(a)
# 结果
[   1.      3.16227766   10.           31.6227766   100.  ]

4)ones、zeros、eye、empty
ones创建全1矩阵  np.ones((a,b))   a行b列的全1矩阵
zeros创建全0矩阵 np.zeros((a,b))  a行b列的全0矩阵
eye创建单位矩阵  np.eye(a)        a阶单位矩阵
empty创建空矩阵（实际有值） np.empty((a,b))  a行b列的空矩阵
import numpy as np
a_ones = np.ones((3,4)) # 创建3*4的全1矩阵
print(a_ones)
# 结果
[[ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]]
a_zeros = np.zeros((3,4)) # 创建3*4的全0矩阵
print(a_zeros)
# 结果
[[ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]]
a_eye = np.eye(3) # 创建3阶单位矩阵
print(a_eye)
# 结果
[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]]
a_empty = np.empty((3,4)) # 创建3*4的空矩阵 
print(a_empty)
# 结果
[[  1.78006111e-306  -3.13259416e-294   4.71524461e-309   1.94927842e+289]
 [  2.10230387e-309   5.42870216e+294   6.73606381e-310   3.82265219e-297]
 [  6.24242356e-309   1.07034394e-296   2.12687797e+183   6.88703165e-315]]

5)fromstring  np.fromstring("string",dtype=np.int8) 
fromstring()方法可以将字符串转化成ndarray对象，需要将字符串数字化时这个方法比较有用，可以获得字符串的ascii码序列。
a = "abcdef"
b = np.fromstring(a,dtype=np.int8) # 因为一个字符为8为，所以指定dtype为np.int8
print(b) # 返回 [ 97  98  99 100 101 102]

6)fromfunction  np.fromfunction(func,(a,b))  func指定与行列号有关的函数
fromfunction()方法可以根据矩阵的行号列号生成矩阵的元素。 
例如创建一个矩阵，矩阵中的每个元素都为行号和列号的和。
import numpy as np
def func(i,j): 
    return i+j
a = np.fromfunction(func,(5,6)) 
# 第一个参数为指定函数，第二个参数为列表list或元组tuple,说明矩阵的大小
print(a)
# 返回
[[ 0.  1.  2.  3.  4.  5.]
 [ 1.  2.  3.  4.  5.  6.]
 [ 2.  3.  4.  5.  6.  7.]
 [ 3.  4.  5.  6.  7.  8.]
 [ 4.  5.  6.  7.  8.  9.]]
#注意这里行号的列号都是从0开始的

5.矩阵的常规运算
+	矩阵对应元素相加
-	矩阵对应元素相减
*	矩阵对应元素相乘
/	矩阵对应元素相除，如果都是整数则取商
%	矩阵对应元素相除后取余数
**	矩阵每个元素都取n次方，如**2：每个元素都取平方
import numpy as np
a1 = np.array([[4,5,6],[1,2,3]])
a2 = np.array([[6,5,4],[3,2,1]])
print(a1+a2) # 相加
# 结果
[[10 10 10]
 [ 4  4  4]]
print(a1/a2) # 整数相除取商
# 结果
[[0 1 1]
 [0 1 3]]
print(a1%a2) # 相除取余数
# 结果
[[4 0 2]
 [1 0 0]]

6.常用矩阵函数
np.sin(a)	对矩阵a中每个元素取正弦，sinx
np.cos(a)	对矩阵a中每个元素取余弦，cosx
np.tan(a)	对矩阵a中每个元素取正切，tanx
np.arcsin(a)	对矩阵a中每个元素取反正弦，arcsinx
np.arccos(a)	对矩阵a中每个元素取反余弦，arccosx
np.arctan(a)	对矩阵a中每个元素取反正切，arctanx
np.exp(a)	对矩阵a中每个元素取指数函数，e^x
np.sqrt(a)	对矩阵a中每个元素开根号，x^1/2
当矩阵中的元素不在定义域范围内，会产生RuntimeWarning，结果为nan(not a number)。 
arcsinx的定义域为[−1,1]
import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print(np.sin(a))
# 结果
[[ 0.84147098  0.90929743  0.14112001]
 [-0.7568025  -0.95892427 -0.2794155 ]]
print(np.arcsin(a))
# 结果
C:\Users\Administrator\Desktop\learn.py:6: RuntimeWarning: invalid value encountered in arcsin
  print(np.arcsin(a))
[[ 1.57079633         nan         nan]
 [        nan         nan         nan]]

7.矩阵的乘法（点乘）
矩阵乘法必须满足矩阵乘法的条件，即第一个矩阵的列数等于第二个矩阵的行数。 
import numpy as np
a1 = np.array([[1,2,3],[4,5,6]]) # a1为2*3矩阵
a2 = np.array([[1,2],[3,4],[5,6]]) # a2为3*2矩阵
print(a1.shape[1]==a2.shape[0]) # True, 满足矩阵乘法条件
print(a1.dot(a2)) 
# a1.dot(a2)相当于matlab中的a1*a2
# 而python中的a1*a2相当于matlab中的a1.*a2
# 结果
[[22 28]
 [49 64]]

8.矩阵转置 a.transpose()
import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print(a.transpose())
# 结果
[[1 4]
 [2 5]
 [3 6]]

9.矩阵求逆矩阵 lg.inv(a)
求矩阵的逆需要先导入numpy.linalg，用linalg的inv函数来求逆。 
矩阵求逆的条件是矩阵的行数和列数相同。
import numpy as np
import numpy.linalg as lg
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a.shape[0]==a.shape[1]) # True, 满足矩阵求逆条件
print(lg.inv(a))
# 结果
[[ -4.50359963e+15   9.00719925e+15  -4.50359963e+15]
 [  9.00719925e+15  -1.80143985e+16   9.00719925e+15]
 [ -4.50359963e+15   9.00719925e+15  -4.50359963e+15]]
a = np.eye(3) # 3阶单位矩阵
print(lg.inv(a)) # 单位矩阵的逆为他本身
# 结果
[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]]

10.矩阵信息获取
1)最大最小值  a.max(axis=0 or 1)
获得矩阵中元素最大最小值的函数分别是max和min，可以获得整个矩阵、行或列的最大最小值。 
import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print(a.max()) #获取整个矩阵的最大值 结果： 6
print(a.min()) #结果：1
# 可以指定关键字参数axis来获得行最大（小）值或列最大（小）值
# axis=0 行方向最大（小）值，即获得每列的最大（小）值
# axis=1 列方向最大（小）值，即获得每行的最大（小）值
# 例如
print(a.max(axis=0))
# 结果为 [4 5 6]
print(a.max(axis=1))
# 结果为 [3 6]
# 要想获得最大最小值元素所在的位置，可以通过argmax函数来获得
print(a.argmax(axis=1))
# 结果为 [2 2]
2)平均值 a.mean()
import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print(a.mean()) #结果为： 3.5
# 同样地，可以通过关键字axis参数指定沿哪个方向获取平均值
print(a.mean(axis=0)) # 结果 [ 2.5  3.5  4.5]
print(a.mean(axis=1)) # 结果 [ 2.  5.]
3)方差 a.var()
import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print(a.var()) # 结果 2.91666666667
print(a.var(axis=0)) # 结果 [ 2.25  2.25  2.25]
print(a.var(axis=1)) # 结果 [ 0.66666667  0.66666667]
4)标准差 a.std()
import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print(a.std()) # 结果 1.70782512766
print(a.std(axis=0)) # 结果 [ 1.5  1.5  1.5]
print(a.std(axis=1)) # 结果 [ 0.81649658  0.81649658]

11.基本运算符
np.dtype	指定当前numpy对象的整体数据, 见下一个表格
np.itemsize	对象中每个元素的大小, 单位字节
np.size	        对象元素的个数, 相当于np.shape中的n*m值
np.shape	轴, 查看数组形状, 对于矩阵, n行m列
np.ndim	        秩
np.isnan(list)	筛选出nan值
np.iscomplex(list)	         筛选出非复数
~	                         取补运算符
np.array(数组, dtype=np.bool)	 自定义数组类型
np.astype(np.bool)	         转换数组类型
np.mat()	将python 列表转化成矩阵
np.mat().getA()	将matrix对象转成ndarray对象
np.matrix()	同上
np.asmatrix()	将ndarray对象转成matrix对象
np.tile()	重复某个数组。比如tile(A,n)，功能是将数组A重复n次，构成一个新的数组传送门
np.I	        矩阵求逆
np.T	        矩阵转置, 行变列, 列变行, 对角线翻转矩阵
np.tolist()	转换成python列表, 用于和python原生结合写程序
np.multiply(x, y)	矩阵x 矩阵y相乘
np.unique()	        数组驱虫, 并且从小到大生成一个新的数组
np.arange	        同python range()
np.arange(24).reshape((2, 3, 4))	创建一个2维3行4列的数组, 必须能被给定的长度除开, 可以索引和切片
np.arange(24).resize((2, 3, 4))	        同上, 会修改原值
np.linspace(x, y, z)	                等间距生成, x起始, y截止, z步长
np.ones(x)	         生成都是x的数组, 可传递三维数组, 几行几列, 具体的个数
np.zeros(x)	         生成都是0的数组
np.full([x, y], z)	自定义模板数组, 生成x行y列都是z的数组
np.eye(x)	创建一个正方的x*x单位的矩阵, 对角线为1, 其余为0
np.flatten()	数组降维, 不改变 原值
np.random.rand(x, y, z)	生成一个一维x随机数或生成x*y的随机数组
np.random.randn(x, y)	正态分布随机数
np.random.randint(low, high, (shape))	整数随机数
np.random.normal(loc, scale, (size))	从指定正态分布中抽取样本, loc为概率分布的均匀值, 标准差scale
np.random.seed(s)	给一个随机数字固定
np.randomunifrom(low, high, (size))	均匀分布的数组, 有小数
np.random.shuffle(a)	将数组a的第0轴(最外维度)进行随机排列(洗牌), 改变数组a, 行边列不变
np.random.permutation(a)	同上, 不改变数组a
np.random.choice(a, size=None, replace=False, p=数组a/np.sum(b))	从一维数组a中以概率p抽取元素，形成size形状新数组，replace表示是否可以重用元素，默认为False，p为抽取概率,本位置越高,抽取概率越高
np.sum(axis=None)	求和, axis=0为列, 1为行
np.argsort()	矩阵每个元素坐标排序
np.sort(axix=None)	从小打大排序
-np.sort(axis=None)	从大到小排序
np.sort_values(‘字段’, ascending=False)	排序,升序排列
np.mean(axis=None)	平均数
np.average(axis=None,weights=None)	加权平均，weights加权值，不设为等权重,例子[10, 5, 1],每列分别X10,X5,X1在/(10+5+1)
np.var(axis=None)	方差：各数与平均数之差的平方的平均数
np.std(axis=None)	标准差:方差平方根
np.min(axis=None)	最小值
np.argmin(axis=None)	求数组中最小值的坐标
np.median(axis=None)	中位数
np.ptp(axis=None)	元素最大值与最小值的差
np.cumsum()	累加,cumsum和cumprod之类的方法不聚合，产生一个中间结果组成的数组,默认一维数组,1为按原样
np.cumprod()	累乘
np.count_nonzero(arr > 0)	计数非0值个数,布尔值会被强制转换为1和0，可以使用sum()对布尔型数组中的True值计数
np.bools.any()	测试数组中是否存在一个或多个True
np.bools.all()	数组中所有值是否都是True, 测试有没有空值
np.bools.all()	数组中所有值是否都是True, 测试有没有空值
np.bools.all()	数组中所有值是否都是True, 测试有没有空值