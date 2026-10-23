+++
date = '2023-03-13T19:26:41+08:00'
draft = false

toc = true

authors = '23'

title = 'HNP'

+++

[TOC]

# 定义

## 格相关定义

<img src="https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230329100039281.png" alt="image-20230329100039281" style="zoom: 67%;" />

<img src="https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230326193037947.png" alt="image-20230326193037947" style="zoom:80%;" />

其中$v_{i}$被视为格基，L的维度是L中$v_{i}$向量的个数,称*n*为格的秩，其中的V就是基向量，基向量的个数一般被称为维数。*n* = *m* 时，则称该格是一个满秩格。一个格的秩是指它所包含的线性无关向量的最大数量，而格的维度则是指包含该格的向量空间的维度。

## CVP问题

<img src="https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230326201029976.png" alt="image-20230326201029976" style="zoom:80%;" />

给定一个在格L中的向量$\omega$,要求在各种找到一个向量$v$,要求Euclidean norm最小，下面是原文中对于Euclidean norm的介绍

![image-20230327093924248](https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230327093924248.png)

## SVP问题

<img src="https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230331113537649.png" alt="image-20230331113537649"  />

## 

## Hadamard ratio

给一个$B=(V_{1},V_{2},.....V_{i})$,存在hadamard不等式$||v_{1}||||v_{2}||||v_{3}||......||v_{i}||\geq{detL}$

由此定义Hadamard ratio:$H(B)=(\frac{detL}{||v_{1}||||v_{2}||||v_{3}||......||v_{i}||})^{\frac{1}{n}}$,该$H(B)\in(0,1)$,比率与1的接近程度，表示这组基的正交化程度，由此可以作为其中一个标准区分”好基“与”坏基“。

## 欧几里得空间

在该空间中定义了向量之间的距离和角度，满足以下公理：

1. 非负性：对于任意向量x，有 ||x|| >= 0，并且当且仅当x = 0时等号成立。
2. 同一性：对于任意向量x，有 ||x|| = 0当且仅当x = 0。
3. 三角不等式：对于任意向量x和y，有 ||x + y|| <= ||x|| + ||y||。
4. 标量乘法：对于任意向量x和实数a，有 ||ax|| = |a| ||x||。

在欧几里得空间中，向量之间的距离通常使用欧几里得距离来度量，即在n维空间中，向量x和y之间的欧几里得距离为：

 $d(x, y) =\sqrt{((x_1 - y_1)^2 + (x_2 - y_2)^2 + ... + (x_n - y_n)^2)}$

其中，$x_i$和$y_i$分别表示向量x和y在第i个维度上的坐标。

## HNP问题的本质

以下分别是ctf wiki与《Solving Hidden Number Problem with One Bit Oracle and Advice》论文中对于hnp问题的介绍

![image-20230331154948051](https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230331154948051.png)

![image-20230331154740160](https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230331154740160.png)

由于在不同问题中不同的应用场景，按照个人的理解HNP问题的理解可以分为以下几个步骤：

- 在有限域$F_{p}$上存在$u_{i}$使得$|f(tx)\%p-u_{i}|<=\frac{p}{2^{l+1}}$,其中l是最高有效位的位数

- t是已知的随机数，$tx已知部分比特位$

- $p,t,u$都被给出 从而求$x$

  整体的过程是在格中寻找一个与不在格中的$x$最接近的向量

  $\begin{bmatrix} p & 0 & \dots & 0 & 0 \\ 0 & p & \dots & 0 & 0 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & \dots & p & 0 \\ t_1&t_2&...&t_d&\frac{1}{2^{l+1}} \end{bmatrix}$

# 问题可解的条件

![image-20230331162400758](https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230331162400758.png)

当泄露的位数k,达到$k>=\sqrt{n}+\sqrt{logn}$时如果有d=$2\sqrt{n}$次及以上的oracle则有1/2的机会解决问题。

# 相关算法

## 高斯规约

![image-20230405152157401](https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230405152157401.png)

> 高斯规约仅仅适用于二维格

```python
import numpy as np
#高斯规约
def gaussian_reduction(B):
    n, m = B.shape
    mu = 2.0 # 设置初始系数
    while mu >= 1:
        # 对基向量进行排序
        norms = np.linalg.norm(B, axis=1)
        indices = np.argsort(norms)[::-1]
        B = B[indices]
        #Gram-Schmidt正交化处理
        for i in range(1, m):
            for j in range(i):
                B[:, i] = B[:, i] - round(np.dot(B[:, i], B[:, j]) / np.dot(B[:, j], B[:, j])) * B[:, j]
        # 验证LLL条件
        mu = 0.75
        for i in range(m - 1):
            if np.linalg.norm(B[:, i + 1]) < mu * np.linalg.norm(B[:, i]):
                B[:, i + 1] = B[:, i + 1] + round(np.dot(B[:, i], B[:, i + 1]) / np.dot(B[:, i], B[:, i])) * B[:, i]
                B[:, i + 1] = B[:, i + 1] - round(np.dot(B[:, i + 1], B[:, i + 1]) / np.dot(B[:, i], B[:, i + 1])) * B[:, i]
    return B
```

LLL条件用来约束约减后的矩阵更加紧凑稳定。

高斯归约的步骤可以被归纳为：

- 对基向量对比 
- 施密特正交化处理
- 条件验证判断是否返回

## Gram-Schmidt正交化

<img src="https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230331101107493.png" alt="image-20230331101107493" style="zoom:80%;" />

Gram-Schmidt算法通过给定一组基，得到一组正交基，通常$p_{ij}$与角度相关

归纳总结步骤可以分为以下几步：

- 挨个去除向量计算投影向量的系数
- 挨个减去投影向量
- 更新基向量
- 对基向量进行单位化

```python
def Gram_Schmidt(B):
    n = B.shape[1] # 基向量的数量
    mu = np.zeros((n, n)) # 记录Gram-Schmidt正交系数的矩阵
    for i in range(n):
        v = B[:, i] # 取出第i个基向量
        for j in range(i):
            mu[j, i] = np.dot(B[:, i], B[:, j]) / np.dot(B[:, j], B[:, j])
            v = v - mu[j, i] * B[:, j] # 将第i个基向量投影到第j个基向量上
        B[:, i] = v # 更新第i个基向量
        mu[i, i] = np.linalg.norm(B[:, i]) # 计算第i个基向量的长度
        B[:, i] = B[:, i] / mu[i, i] # 对第i个基向量进行单位化
    return B, mu
```

从原始向量组中选取第一个向量$u_1$，然后对于后续的每一个向量$v_i$，我们找到它**在前面所有向量张成的子空间上的投影向量**，然后将它减去这个投影向量，得到的向量$u_i$与前面的向量都是正交的。

## LLL（Lenstra-Lenstra-Lovász）算法

> LLL算法是一种基于Gram-Schmidt正交化方法的改进算法，用于在高维空间中对基向量进行约减,能够有效地将一个给定的整数线性方程组中的向量组进行约简，从而降低计算复杂度。算法的核心思想是通过连续的基变换来减小向量组中每个向量在其余向量张成的子空间上的投影长度，直到满足一定的约束条件为止。![image-20230405155655177](https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230405155655177.png)

以下是LLL算法的主要步骤：

![image-20230420232045946](https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230420232045946.png)

1. 初始化：给定一个正整数n和一个n维向量组B作为初始的基向量组。

2. Gram-Schmidt正交化：对于给定的向量组B，通过Gram-Schmidt正交化方法将其正交化，得到一个正交向量组B'。

3. 基向量的长度排序：将向量组B'中的向量按照长度从大到小排序。

4. LLL约减：对于向量组B'中的每个向量$v_i$，将其与前面的向量进行一定的线性组合，使得得到的向量$w_i$尽可能地小。具体地，对于第i个向量，将其与前面的向量$v_1$到$v_{i-1}$进行线性组合，得到一个新向量$w_i$，使得$w_i$ =$ v_i $- $μ_{ij}$*$v_j$ (0 <= j < i)，其中$μ_ij$是一个适当的实数，称为Gram-Schmidt系数。通过选择适当的$μ_{ij}$，使得$w_i$的长度最小。这样一直处理完向量组B'中的所有向量后，得到的向量组B''就是LLLM-基。

5. 判断：对于得到的向量组B''，判断其是否满足LLL条件，即满足：

   a. 0.25 <=$ μ_{ij}^2 * ||v_j||^2 < 1$，对于0 <= j < i < n；

   b. $||w_i||^2 \geq (\frac{3}{4} - μ_i^2) * ||w_i-1||^2$，对于1 <= i < n。

   如果不满足上述条件，那么返回步骤4，否则终止算法。

```python
import numpy as np

def LLL(B, delta):
    n = len(B)
    B = np.array(B)
    gram = np.dot(B, B.T)
    basis = np.copy(B)
    gs = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            gs[i, j] = np.dot(basis[i], basis[j]) / np.dot(basis[j], basis[j])
            basis[i] -= np.round(gs[i, j]) * basis[j]
        if np.dot(basis[i], basis[i]) >= (delta - gs[i, :i]**2).sum() * np.dot(basis[i-1], basis[i-1]):
            continue
        basis[i] -= np.ceil(gs[i, i-1]) * basis[i-1]
        while i > 0 and np.dot(basis[i], basis[i]) < (delta - gs[i, :i]**2).sum() * np.dot(basis[i-1], basis[i-1]):
            basis[i], basis[i-1] = basis[i-1], basis[i]
            gs[i, :i], gs[i-1, :i] = gs[i-1, :i], gs[i, :i]
            i -= 1
            #验证条件
            if i > 0:
                gs[i, i-1] = np.dot(basis[i], basis[i-1]) / np.dot(basis[i-1], basis[i-1])
                basis[i] -= np.round(gs[i, i-1]) * basis[i-1]
    return basis.tolist()
```

## BKZ算法

BKZ算法是一种基于LLL算法的改进算法，其主要思想是使用一系列的基向量变换，将原网格基转化为一个更加优良的基，用于处理CVP问题和SVP问题，其主要思想是在矩阵的列向量之间应用一个“剖析函数”，将矩阵分成一组小的块，然后对每个块分别进行基向量变换，使得每个块中的基向量之间具有更小的Gram-Schmidt正交系数。

步骤如下：

1. 初始化：将给定的整数向量组 $V = { {v}_1,  {v}_2, \cdots,  {v}_n}$ 转化为整数网格 $L = {span}*{\mathbb{Z}} { {v}_1,  {v}_2, \cdots,  {v}_n}$。
2. 选择 Block Size：选择一个正整数 $k$，称为 Block Size，用于决定每次 Gram-Schmidt 正交化时需要处理的基向量个数。
3. 对初始基向量进行 Gram-Schmidt 正交化：对 $L$ 的初始基向量 ${ v_1,  v_2, \cdots,  v_n}$ 进行 Gram-Schmidt 正交化，并得到正交基 ${ {b}_1,  {b}_2, \cdots,  {b}_n}$。
4. 进行 LLL 变换：使用 LLL 算法对 ${b_1,  b_2, \cdots,  b_n}$ 进行 LLL 变换，并得到新的基向量组 ${ {b'}_1,  {b'}_2, \cdots,  {b'}_n}$。
5. 进行 BKZ 变换：对于 $1 \leq i < n$，将 ${ b'*i,  b'*{i+1}, \cdots,  b'_n}$ 按照 $k$ 个一组进行 Gram-Schmidt 正交化，并得到新的基向量组 ${ {b''}_1,  {b''}_2, \cdots,  {b''}_n}$。如果 $k=1$，则这一步就相当于 LLL 变换。
6. 判断停止条件：判断 ${ b''_1,  {b''}_2, \cdots,  {b''}_n}$ 是否满足特定的停止条件。如果满足，则输出最优解；否则，继续执行第 5 步。
7. 重复步骤 5 和步骤 6：使用新的基向量组 ${ b''_1,  {b''}_2, \cdots,  {b''}_n}$ 重复执行步骤 5 和步骤 6，直到满足停止条件。

```python
import numpy as np
def BKZ(B, delta=0.99, blocksize=30):
    n = B.shape[1] # 基向量的数量
    m = min(n, blocksize) # 选择一个块大小
    B, mu = Gram_Schmidt(B) # 对基向量矩阵进行Gram-Schmidt正交化处理
    k = m # 当前正在处理的块的起始位置
    while k <= n:
        B[:, k:m], mu[k:m, k:m] = Gram_Schmidt(B[:, k:m]) # 对当前块进行Gram-Schmidt正交化处理
        u = B[:, k-1] # 取出第k-1个基向量
        for j in range(k-2, -1, -1):
            # 使用Hadamard变换将前面的基向量全部投影到第k-1个基向量上
            u = u - round(mu[j, k-1]) * B[:, j]
        # 计算新的第k-1个基向量
        B[:, k-1] = u
        B, mu = Gram_Schmidt(B) # 再次对基向量矩阵进行Gram-Schmidt正交化处理
        norm_B_k = np.linalg.norm(B[:, k-1])
        if norm_B_k < delta * np.linalg.norm(B[:, k-2]):
            if k == n:
                return B # 所有基向量已处理完成，退出循环
            else:
                k = k + 1 # 处理下一个块
                m = min(n, k + blocksize - 1) # 选择下一个块的大小
    return B
```

## Babai最近平面算法

baBai算法可以用于求解格上最近向量问题（Closest Vector Problem, CVP），其基本思想是将CVP问题转化为求解平面图形交问题求解。

以下是baBai算法解决CVP问题的主要步骤：

1. 网格基的构造：将问题中给定的网格基表示为一个矩阵A=[$a_1$,$a_2$,...,$a_n$]，其中每个$a_i$表示第i个网格向量,也就是基向量矩阵B。
2. 前向回归：对于每个网格向量ai，构造一个新的向量bi=ai，然后从左往右遍历bi的每个分量，将其调整为最接近整数的值。例如，对于第j个分量，将其调整为离其最近的整数，然后将剩余分量的值减去ai[j]乘以调整后的值。这个过程可以使用LLL算法中的前向回归实现。
3. 平面图形交的构造：将所有构造出的向量bi表示为平面上的点，并将它们相互连接，得到一组线段，即平面图形交。
4. baBai算法的应用：使用baBai算法求解平面图形交，得到所有交点中距离原点最近的一个交点，即为最近向量。
5. 后向回归：将求得的最近向量表示为网格基上的线性组合，可以使用L算法中的后向回归实现。

> *Babai最近平面算法仅仅在欧几里得空间中适用

![image-20230331151351515](https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230331151351515.png)

```python
def Babai_closest_vector(M, G, target):
    # Babai's Nearest Plane algorithm
    small = target
    for _ in xrange(1):
        for i in reversed(range(M.nrows())):
            c = ((small * G[i]) / (G[i] * G[i])).round()
            small -= M[i] * c
    return target - small
```

# 适用场景

## ECDSA中的应用

### ECDSA

![image-20230406100123080](https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230406100123080.png)

其中的e就是nonce值

### biased nonce attack

biased nonce attack是一种针对ECDSA签名算法的攻击方法，其攻击目标是私钥被保护的情况下，通过构造特殊的消息和签名对来获取ECDSA签名中使用的随机数（nonce）。通过预测nonce，攻击者可以使用已知的签名和消息计算出私钥，从而破解整个签名系统。

### 2022LINECTF Baby crypto revisited

> we could still capture the first half of the nonce, which is 64-bit this time. Now please help us to find out the encryption key again. The victim is using the secp160r1 curve. The following is the captured data: r, s, k, and hash respectively. 

```tex
0xe6b7c5a62d08e0216e1e7ed7948c96b74c0be9cd 0x49e1050393f885117de74e7a02d1091d67faa3d0 0xff07bbee67c3ab910000000000000000 0xe91f3200a87205d18a97bdf3bb3027c9f532c8a4
0x7e7b86c8624c9b597131bb883053b1856527a5ff 0x7787b9157fbbaf178ed091b23ce30b2e1ccf9abf 0x882f44f29c56aea60000000000000000 0x37c9f0d06570b0087430b9c66372e385839bb348
0x7951eb8b6ef6ebd080c0171252c53b40fd4ac3b5 0xe70efd784e8a5a35ebd875c4df23132324946e5f 0x842f1342234670730000000000000000 0x7635013447c4b0e7c637dde0b9f8f2eed2cda796
0xae1f8afabb6c971626e8616f30dc0781dee744ae 0xd836a959f2e963291c572a261081ada95ff8a3a 0x37261d496392b4140000000000000000 0x2e7684cae69cd153426acc333bace2c6a294ed4c
0x7b2ad17514e099820de955acf788a0820ce84d0b 0x8d55725ac8252cdc7b65ce9eb12fb87961ec0026 0xe1885611d2d677ba0000000000000000 0x4df56d8d7d972857e5d24d0b18eb0d5738cec1e0
0x4381ac639a53c05bbdd79e936edb1a9aad643a2e 0xb0a40dbd60ff868d233b4446a61592043eb3ea8a 0x780b62dd774107aa0000000000000000 0xf9a118138aa3916a469b0cf37a04f6b763ec978a
0xdac404ce56ca251e570de84c34bf53884c8b7f76 0xc99e802da4c9718c713ad53b616e25a6bb1a476b 0x6c5c4d7eacb9103a0000000000000000 0x926ca7ee88e96e6aed71e213e3aef1e3425583aa
0x6d51d4f6f0bb10d5d07a82d59b1628876ba4450d 0x5abb936c5ef30cb9bcf3631242913678a473d177 0x72d85cc292b16b4f0000000000000000 0x97e289e5c89dd74bb626b52ec20daa8436900d6e
0x49b0ee21550f81218140759ab532487cc061d82a 0x780667331bc3195a87e5134b7751b58dd629f59f 0xd0c890b60ce8690f0000000000000000 0x83864c4a2a8ef1dbe12f86c1fe1251cf4ea32985
0x4bec5ac8f1525aad358a7350ee3601830a3ab681 0x8756130b26ba2270567245a285eefac5bfbd5bba 0xe23134ac39d15b820000000000000000 0x7eac779906cf18583ef063e3914c7d4ef8613d96
0x9d1cb4a8585d7b06d2a0e04690e02ed8ae64ff54 0x23932777064b826c7c8af984026a85cbe2b08ceb 0x3b90e13716413f700000000000000000 0x1b26f574b59569df40e0e481d66e297fcde3c3f6
0x86ecc400146f0617e896bdccc4382739603b6f5a 0x9e4afd9491b42c090a36893d1df061cdf878fdda 0xb43c4e3c3c65c8a00000000000000000 0x8ab5f4fd53637009cd513d1093ed10eff4dc0e50
0x4428dd295aa3ece3ad30014409cfb3507a101d8 0x977d7d7a6f5ba357e95abc318390559afc07714 0xa4be8fe0359fdc820000000000000000 0x145a2b497aa9e6992346114aca84aebd04cb09e5
0x8ed2e65f209a9a0ac0757c713df5a34423686cf9 0x8dd527f779f237b8ae8123054b4e38e1e38593d5 0x8d5b6d0b2d9130510000000000000000 0x85fe335da54392a4ad496949264bc7386df81646
0xfdf493f3bb2c0463c41285bf570433c2977e156d 0x4a0839bdbfcb47c729a6102a59c9d25047e9f0d4 0x8da6918184b9c850000000000000000 0x596aefa36104603e5fb232a70726ee1808f01164
0xba3bbb3e3bcb9496ca7963d3dfecab171c45e6d 0x99d555ce30419c959be264ae630f129ab5e1b7d 0xad596077f977b1290000000000000000 0xff931a4d4c71440bc5c646cad52ffde88b2b38e0
0x8ae77edb32ac3d75ea6be8ee91ac9a7f45645523 0x7a40b474cbcd46d9448b4146b3ee0de7b63c5e25 0x602062c9b690d6140000000000000000 0x1cb4b70fa07ab7308a7ddb48c2587e35815c72b
0x3806e9952c000dabc5e4a86ce8173c7127e3f41c 0x239d049b92ce6134ebb95c37747e98a8298a88b4 0x9b431f07c92b58f70000000000000000 0x64cea232f3064ac499711d8f6f1d103f6ceacd6b
0xd6e3dee5c6a167149913f316cc18532b8129cfd1 0xad5608b9397dfdde5fc0f6b9d231c7772447c072 0xf66ba6dcdcb464090000000000000000 0x521390dd48004a53eaf5be78a096a4b823f5d4af
0x257fc47aeeb5965ee5c61abd9bd31a554ac91180 0x85c24d04de023897f2cc01ccaf6888ea6449deb6 0xfa0c651bd5f61f2e0000000000000000 0xde81f2aace5e3327a6cfb265f129b45063e23118
0x788fa1d05416d6e04f25c77b3d09144dd556a59d 0x50be06a69bc98afa1376bece9b7da34b4f89ea98 0xd45b63777d3caa350000000000000000 0x873fec7ae37daf5d62fb379325058b55e875ffb0
0x437cf2db39f9fe83260997a4710febb5e88662da 0x61addc6a7bfd5d794df4f2b5caa045f86de743a3 0x7a9c32aa743d21640000000000000000 0x538c6e0b0f363372e4768b0d5fb363cf22b2c67f
0x64df7a272b47aae82dbbf7a93416d255777c41e6 0xd5d59703e5fc0f9b0acbd87be8cf43a78a026787 0xdf9faa705f6e1d790000000000000000 0x5590f75b5d65d44dcd82f741ce7eb5e8c47451f8
0xe89ba3e95c9fee54f5659c8ab098fb8f6d956fdc 0xf79c5519709988561774f33fcb28939b7bebe878 0x897c90fc2d0892210000000000000000 0xcafa5838bea1ef0222044fa50f1c17eb200a1758
0x5b1f7d123b12ec8280280c4e21fa058d738eda30 0x9df50a51e8d3785c1e5dd58b766b54d81f415c4f 0xed3a6323c982c8770000000000000000 0xd1c67285245b4a7ffdda4e6ba7f18300927b99e
0xc9ac314cf3c9da5fb0f23384739a9fc1ef0cc640 0x3bb2e7feeb88e016bfb6e6e1262315c1c0493465 0x3a8a9d9f96db142d0000000000000000 0x1f2a37293f964a8508bb08f033f96f9e36390eee
0x583fdba4bcc386ef8d169b73e0bf996055179cae 0x32be93a2984d55aead8e1a76f81372d4824de612 0x20a83241c9aff2350000000000000000 0xaf0d64c0313614970406c6ca025b981d1d1a23a0
0xed7d2c6be42d291953b6c199946f4a342d18baf5 0x9a353f6ad7b12fad1015f728c7482411c9378f01 0xbaf7d7a94683c6fb0000000000000000 0x4e7892b445e3516021fa6f806d307df14915526a
0xd78f0ea703167bc0113436d8d626b2a135d29dee 0x86f9f56551e019cad8cc1c418b4db34b194f3db8 0xf8c49fe987b3ef470000000000000000 0x12960c5c53da52f6dd3d07c7e430980b9b86cd15
0x34d3ddac6f5537c7cf52494b7746c9a0fd60edaf 0x8a9bc5a06753324019a9a1a25ab7d50d053213f9 0x84bdabdec83278430000000000000000 0xefcf6aa28ecc6f9b98f89fe3afb4625a74b22626
0x853f3e3abdece6b0c76d618cbd7d48f5f5265ba6 0x4a56c6b0a8f25c855d67d38e4e517bfb7b22f135 0x39bda4b462ebf9ac0000000000000000 0x51e51a8d404c978b404d6b2e413f9c51d590e0f2
0x77f1eaa8661d0b71130f7eab92ac69e773e78f61 0x188a40b525364648a1024ac473989db9425a81e9 0xec62902e499f85a70000000000000000 0xadab031c2bf87d93e185dd17baa2a0f8985e9f62
0x6990ee35c1c8f654c88eb48ab4690b1204f3af36 0xb78b8ba2838245bb378411173c67ac899f7205b5 0x9f24a0b08009a6530000000000000000 0x3ac390469826f7d4da428ccfcb436570f963d5f2
0xa5502fc425956a2291547409504502770dc201f3 0x35697262d72fc87ebc41d3e1d8bb906344943d0 0x3280618ea81054060000000000000000 0x1724f2d5113739c34276d13fd9fc3629cf13dd3
0xf1bc123ddf05fa7ebcf6860e682afec86e0c4a57 0xd26fed816fbb6628d126f7611c988a091d1e9bf 0xe1e42eb6182809400000000000000000 0x5e36883256500ff7af465001d4851c6e2690b623
0x670ff55021a66754e7c57809653120f584bc3bf8 0x7c3e02264b92b9a1bf7e30c083e123f66693e373 0x28a30fca65191d1b0000000000000000 0x99f98876f8d5f573d854469b61977b0acd8a2bd2
0x5488d7a1391774a010959ddb6ebb975e0c37d278 0x86fd741c0494d137f70fe1ebe83b55642b07f65a 0xad5a6bd145c3e2f60000000000000000 0x35fd04f7c0d5b17370955a096381af38d6290caa
0x686e908241545f4e33ef9dc93ce51f08790f0e50 0x113f451f7892d9fd277515098968332ead3eccc9 0x7c880dc9cf759d110000000000000000 0x65d822fd2ff3fa621d054c7c25b7ebcf9c0531cd
0x1255e93e812cb1f70224176425c35b899517266f 0x2f302814aa0b0520c5f2f78bb816024d891a6717 0x53b0bc6e7eb074af0000000000000000 0xd92e3db6229699541227bacdd2e3c0f3e2be4e90
0x4452646403e187b3f8f77a2dd98e35b92844f602 0x78fb3383e0bce43facfb89f883332afd17feb1ba 0x46d4e3de03f120a50000000000000000 0x51ac3a965f4df2d17ac1dde16108478015e373d0
0xf80a1b4add13ea59eea34a586a63c0a703b76243 0xc3424e7004ae49ed9203ea8994448ce5427944a5 0x3b7c93aabc26062c0000000000000000 0xcbe7ef3101a3a45c6dcb6cf5ddd008f29b471c64
0x7338aa76ec3f68f2904b6632b10623e0e322061c 0x53c2a336ce2ffaf849302f38e845f6c6a065242f 0xde7c8a452ba7adc00000000000000000 0xffdbfc9ff1030263c2e6c9dd8ddcabd990b5db10
0xa382764a124e32315cf74abdfd87c1f0fb3dfa16 0x6711a1796b18fd2e2d8bb38ef58381aac4151b77 0xbdc65a8ef35707790000000000000000 0xd24002afb6687ef92d25020eadd8fd49080e19f7
0x870eb73ea135b19f826674231e9329f4e4424ed1 0x3b0d83e7be7727472b28e2ee368fdbd36ca6e9ff 0xe4624c08a1ddc0ed0000000000000000 0x4a9f1dab7da93bb499ecc3483e658a4287c1f81
0xd2fd9af58203e96b4f7a29c62684c6eebcf4a2ca 0xe95563d5c279720768d2c70d2d113727e6aef177 0xd9b040dda650ffd80000000000000000 0xc5a220169b4df1b2e8f50f75229fb7eab14c0890
0xdab18d09e0f396509fd904c3bf7b450eda39a518 0x79afa09ca5103adf122378a93502adf6d7d05450 0x3826a0ba7ca02eb00000000000000000 0x3414a64e27e015d5fad46c2a533e0778f3b82cf4
0x7b467a055893df2de9bfaa548d8ef0133920e309 0xd3cb6d5cc9079a5616700648d43cab650a403f5c 0xd01a1eaafb56c2360000000000000000 0x2a65254571600122b31b85334c34e554f5e04295
0xd5247e6863813b33ad850f73ad3fd5e2cd593fec 0xcd63a701b42279c7ee39dab6cb680298402c8287 0x5fa192029d8470b70000000000000000 0xb08da686395c91eaa2243b3fbf13ca39eec2d452
0xc2f72a004238c1b332f0116756a74ce17cbc821f 0x26f47b05697d0436c4e6d3e44142beddc128f99c 0x3016ffd01a1c5f8e0000000000000000 0xc693ee8c1eb51babf6dbb6f83447a5626d3cdedc
0xda237fb136d2f59f94a25874157052c1525dc4b9 0xebed90278faff7a82f688a760c45ccb08c483e9 0x39083d2431bc3510000000000000000 0xb40f9600a1aba3c71e98d6991ce83a1f53d22e8e
0x54e051ed46868c35647488d1775166f8a2bc1eca 0xb8aa523c78db57eb5d3af17ab4586e4b5d433ee8 0xb44e15d3a201c01b0000000000000000 0xb6c6768be8e9c5bb1eb80c3f14e9c9ee1a9a58f4
0xb688f4b2e78bcc615b66fce199e66c9b5dc0b0a6 0x274e79591003aea61214b2b66a090d6f3ceb163e 0x86136affdcad79e70000000000000000 0x5adb65f35b5f458e83ebc33d8b112e2f7371608f
0x11b086851c41e8efd035dbe45611558319281639 0xa6662f015f062cd6acacaa82300817f2f60e786e 0xbc5343fb25aad5c40000000000000000 0x1bc98491891cbcd46f48da844c0ef937270da02a
0x629c7e28d77a627d996ef6232e561602c0a7f45 0x62564bd9e56b2cd28859f489b397d413b7f5200c 0xd0e592bae1525c7a0000000000000000 0x7fdef748ed2e22bf063774b1f28e23501fbf108b
0xafa0d19f556032c79afe2d83e4c098a065fc004a 0x21037b0796f2504a77b694f68ca364e4b2fb6533 0x2d7f209ad5fa84c60000000000000000 0x530862133947108ea86a83d44939845182c51bba
0xe40cdfc61602daddc5152bb10c862b85783aa703 0x6381388c135b497f8b39ddc2751528c128de97fb 0xb48a071d4a45de690000000000000000 0x10139d0c52f2cd4c1f70b481693826b03bb8e363
0xce20b275201922d29cd90a84217134c5f9c5e93b 0x8f867c323d6e56786a514ae1d43a83cf1b2f0aa6 0xd13db177aa94e72a0000000000000000 0x2b2a35c4cb24f1237aee91e960dac2d80e3597ca
0x63f691867126d15cdac5bbcf5f6ad7779caa3a42 0xcb188c080cb07cd92cf7fe7329f7e2583493ffe0 0x2336c28d1173ea820000000000000000 0x720b964116e5546c65a42dea2742265a03e2c54c
0xe35fb5ea0819382432d44a907eabb367b80c9b8c 0x893ca5dcc044a545feb505f1220bd5741576e1f 0x6972ff4e322270b60000000000000000 0x8b26bcf5ec720e8be3554d39f276ae2097d18be8
0xac73025a8706374c46ace2cfbd8dace3d07ca021 0x6b5c98934731276252e40c6f747729838f02341d 0x9993b2bc68d13abc0000000000000000 0x16c98b895b826cd05f7f49f87f42b4373bad4d7e
0x120961f077341ed751b4af1df17a53ac805b887c 0xe060848d198322e910937040a79f4092a372d926 0xf0f2980c89e607260000000000000000 0xd5480a73235a74efac49b4817b6b4057a3a779f8
0xffe7b85c9583f89b8a175b5dc44b5715ede3da93 0xe1a27334c73c67ee8bb283fbb7862429c8650a27 0xf449734df37c6ae70000000000000000 0xab3cbe4f907766a659cef41504fa4b3859996e03
0xbd059cf154f66bf6d96897c8342994124253d9c5 0xac3cde83dc44daad35193779defac4614efe093 0x7e9f05472c27bd940000000000000000 0xee86ed8897dc982fa67e57bb1d4099a0478c0807
0x45219290f348b5218ea4f0fc0ddccf2c1c7c4ae7 0x1c0537ebdc28618f81c08fd307e9abe08b161e4 0x9adc2b1247bfa30c0000000000000000 0xaef33971ec91a7a39f1f8a4681cc4d58f993e887
0x3ee7b29904619b72dae7ef3df040b75bf88f4b39 0x5fdeaf1d0801129252800b38bde5bae55affe30e 0x8e97ddc7eb75d24c0000000000000000 0xed6285792423062edac86d5b6fe4d7e73ad18006
0xf1622ca3a2570522dd8ca99234f4e28d30b0d4ae 0x8f8c73823510adb18bb62b3816204b8458544f1c 0x6b3098217fbe71e0000000000000000 0x30a1b20fa0b397ed0ce1a0ca7d4eecea74b1fbc5
0x13fb9073eb91ff8d46f14360b8aa77e0c1be4d9d 0xa1dcbd89559a9f6744e33bc0a0760c2ba78ad87e 0x4d3aa977e0fed920000000000000000 0x83190f142f789d9a6b7008645129e6078ea9bdb6
0x3d7ab5256052a2c3c23460986dd95d728a67c6ed 0x263be5b1c7f53cea5f43cf3706fa7dfe36699c32 0xdbb1bbf5f0a5fa050000000000000000 0xbc539b98057c1c637d33463a118ec72ee1b6d436
0x8d99ec935116d6b177b2f08499736a6e883b6417 0xd9a9a4caab6269fc43eb411dce43ef9a4ed95ab9 0x52e15c12050acc9d0000000000000000 0x9968c99dc062c595e15c8cfcfd5f675b96c90b4f
0x86289261b2d9d7a51c56a8a3d6e0cbce58b8a724 0x16d635ce95565d63f6b81c0b863cc8d95759a37c 0xc8b58b95ac93c0110000000000000000 0x5149b823c5a0b3f7220233de3ab97a14ed73444f
0x9c4f63ec50460526b8a1fa07b14ec3cd1e6e3237 0x2108afa213804ad90d1364582d2667ac44390cf2 0x71363c809180b1590000000000000000 0x2f69376425cab83532e27e0e8b28c7081ff16ef2
0x3b3f24f396abf45c73981b795d8480da52193c87 0x33076e0bf25bf487a866a00ba0cb7f6c7e19a22a 0x66dbde72c010e6550000000000000000 0x6b0e91b86908f91eeb596f660597db81c7cf8db3
0xb729ea459e7f54d8474136b50397ebbbe2dcfda2 0x8145192e5226c1511b8bc1e63433e6a4fa1bc87 0x8f99f5686c12be940000000000000000 0x933873f8119224df31a1678f3d8b30a3e24e501d
0x22fbc9a734fc61cdcb503452c4e823b33a191813 0xb9e3022ce38256af3aed4d4a6c12aa7b9b3b956c 0xa7e48509accf5eea0000000000000000 0x2298ef29bb513fc794fda3a659b64d173a68c829
0xef6ce3760374739d6b32e8072e411f46f73fc0ca 0x27c9603fcb1f28519b15ed5f64d95a160840e32f 0x39d7b0b6ba451e110000000000000000 0x176e62b08958b099f1db049e4c5ce410ebcc71b4
0x533c27b591423f3d7fedc96b85ec103289cccc 0x1b6d50ff1f21a294c1db33a554d3edf7fb2514a5 0x997314384fc664c0000000000000000 0x6048177f7e75d654722a862dd1b0796d46a8876f
0x54a78a87181cfc2f59d63f1eafd827863916f933 0x454d75207492945c8b542758d1fa92d843477e09 0x86bd8fc8c64226520000000000000000 0xecb9696dc402db8f780f75274a8141a4b54de8b7
0xfce33429e9494aa8aeb3e66c4e01ff93b155808a 0x5f9e36435d87337500f951833ba51361fd64da02 0xe5b6a81bf343c1ad0000000000000000 0xb2d0a0496fcaa68593af04cf5571f99d1cd408a5
0x6977d5dc3ac449e022175a61f095cb54365e72c1 0xdb4fdf9d86bbed27d1c2803884134aff2b0bf8c5 0xec04900fa28845b90000000000000000 0x2fd1eb2d9c928d64b5da3646f6b21a5b9812496a
0xa7226c444c51fb597abb5205300c00e1f0f11929 0xbc92c3c94536d45336b9455f55bd1d8d1e088007 0x370bb04f631a82860000000000000000 0x65948f2ebfc7c0c4e156eb4b63a134aa2843c030
0xeaf2dac105fa53486be1e5125de9e5a91f8bbabb 0xbda4419a3801880df6bf147fafe6c53d739b1fa4 0x3023fad161cf41bf0000000000000000 0x76b1cce00ab40da7ee529825db2d40ce9065dfa1
0xfab0b8d5df57f88528d525b775a505fe0e5343f7 0x441de70f5b659952f1520924190f961fe03bcee3 0xf3638f7a103530850000000000000000 0x4e873c30a1b7d817525c487872d23d4835514f3c
0xfa713b69bd51c5d2fbbb449e1a81ab49e7223b41 0x5513b96e9e4abc03ee371fe9c651bd571e1a0f80 0x5b9e909967e524bc0000000000000000 0xbfed8332e16807b46aac7f514dc9e0d855b6d99f
0x3a568f652c65a2813f6f0bf2e5fd2d309064e2c5 0x7e97233fa957513a44df6cd523a06ebe8d03fade 0xa091d9a2f9124c6d0000000000000000 0x285b20e0028d4bb6346e592181411d0653e42551
0x207b28b1447e900014ab4ef4870d754089d37e78 0x5aec49288e0eac889ea91b594fc9415696c7d8be 0xaff4138f7fa677c50000000000000000 0xec16f04c2c6920f5272096ce99a41499cd81edc
0x2687452a3308fa856640a6b06e165898572e3eda 0x7bde5d206ed6a30923411737aaf5a690c5c4e5a3 0xd5f0e5b9da2596fe0000000000000000 0x6926630cdd5580e3379d01ae82cb7f18c2e8d9
0xff3564911d6b151f4f95a418a1e6ef5643e0ca0 0x5a42a1a72e04c9e00383ff7e19fb397b66b3c9d1 0x9cb7334b764929670000000000000000 0xcfb1b883e80fd73c92b59c3d234d71652b7b192b
0x3b98f40b90e245b1010cd309d3ba83a6e9e0b12e 0x8e51a9175b9031ec7dff0b8a6bfaadcb242e798a 0x6456cfaef9bab78c0000000000000000 0xf7740cac947e576b5747caf29ef8b68ecde565b6
0xef6bb59cad74eeeb916e5938f2e17e943cc2d8ef 0xb1487bf60cc6ac16ae0e2a9487efe453b322a322 0x6b38768240b566340000000000000000 0x87a4faddf0a248c64d1a2b10b49ab4556be915ef
0xb9f6c1f51cc2f8b3a41f160f77f0ca5ed9643a4e 0x2a55fc5b6959fc38e8e9fe70223e62cd79f48ee6 0xcbf740b243f516a50000000000000000 0x306a13deec76212f9832f1651b04114cfa1f995
0x1adea3627bca45da84ae024eed02d1cd13bd1652 0xd864d4e7217dea861b7bc930e233bae0ee2fe41a 0x38932ab70c57f2270000000000000000 0x33d61e2be5e0c5021a13aee3332d3b11a436153f
0x163937e75d099e0ac3f94c1d076b14bb44b97fe 0x578b0cdbe8265b81656ff6581c22ae6ebd54a412 0x9a594647712438410000000000000000 0xa04387096eabec7af4759dcbe2723e2869294c94
0x71c2b00739da1d93a442adced91218aeeb60dc24 0x6f776172a25fa58c411dba7734e22aff5b078311 0x801d6179e974ec680000000000000000 0xdd679c3251849627621a065ee38217238973a51a
0xcc266f10b502a00f14d199e4fe6eddd130036c02 0xd92ffca814ddfd379a29b4f9bf27a78b941732ba 0x9f761367db7f2c810000000000000000 0xbdab36d4fa755e11ebe8bfe8b8eb38d852a55ffa
0xc349aae790293aeaf32524c91ec28a9aac1037b4 0xcba010ede919bd26bccbe24ac191e2e84965d854 0x762202ca77125cff0000000000000000 0xa7af0045a148bda9174c1f1a02a59a33e58b9064
0xc0b52d090295c7324b7a84d54e99329843b0bb21 0xdddd85f5477bf7a311b1fec5edf5bdde9c5b0bf9 0x6d4d198be1c7c54e0000000000000000 0xabe2b9ffdb26e12cc5894a97a2b992e271b2beda
0xd0602fba8a8af9e8717707b1af7d22499bc83b43 0xb5d7a3613e4cb2f4bbc9d129de7930e08400bd86 0xf15d4787c81d06a40000000000000000 0x2eb6451d26dc07502bfc6aa2933903b3ba7a9e1e
0x7be000d6d05c7df51bcafaa3bfb4e14bc6e4e081 0xe73f5109496e115ed57e18a6d9bf4907999b11e6 0x33764502c1bd3dc40000000000000000 0xe8745c280022281dd127be1c403a8d504375b739
0x7762f4929d5c0c3b6d2cdb0fd454df1f9f84a90b 0x5b22fa11850d966e755017746ba6c1089d1fc767 0xfee73df91963e3070000000000000000 0xea7d33183c151eb74f1a888c1ed0373e58badbee
0xbefd0c44fb52bfde7d7e4b3362d8a8f0fa7e18fd 0x302080531bbfd8c98d19f402b56415d962cc145 0x6de2a164d855afe90000000000000000 0xe545a32fa33bd935682a1fa376c272c1ac363583
```

#### secp160r1 curve曲线

secp160r1 曲线是一个椭圆曲线，也称为 NIST P-160 曲线，其方程可以表示为：$y^2 = x^3 + ax + b$

其中 a 和 b 分别为曲线的系数。对于 secp160r1 曲线，这些系数的值为：

```
a = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7FFFFFFC
b = 0x1C97BEFC54BD7A8B65ACF89F81D4D4ADC565FA45
```

secp160r1 曲线的参数如下：

```
p =0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7FFFFFFF
n =0x0100000000000000000001F4C8F927AED3CA752257 
G = (0x4A96B5688EF573284664698968C38BB913CBFC82, 0x23A628553168947D59DCC912042351377AC5FB32) 
```

其中，p 是一个素数，表示曲线上的点坐标的范围；n 是曲线上基点 G 的阶，即曲线上最小的正整数 k，使得 kG = O（O 表示无穷远点）；G 是曲线上的一个基点，它是曲线上的一个固定点，用于生成公钥和签名。

在 secp160r1 曲线上，

- ECDSA 签名方程为：

​		r = ($x_1$ mod n)其中，r 是签名的第一个组成部分，x1 是曲线上点 P 的 x 坐标。

- 签名的第二个组成部分为：

​		s = ($k^{-1}$ * (hash + d * r)) mod n

​		其中，s 是签名的第二个组成部分，k 是一个随机数，hash 是待签名的消息的哈希值，d 是私钥。

- 签名的验证方程为：

​		$s^{-1}$ * hash * G + $s^-1$ * r * Q = O

​		其中，Q 是公钥，表示曲线上的另一个点，用于验证签名。

#### 题目理解

nonce剩一半 ,secp160r1 curve曲线则p可知 题目给出了 $r,s,k,hash$,其中$k$仅有一半，在hnp问题中将其看作$msb$

存在公式如下：

- $r=(x_{1})modn$	
- $s = (k^{-1} * (hash + d * r)) mod n$

其中 $r,s,k^{'}$已知,令 $k^{'}$为k的msb,$k^{'}$为64bit

可知：

- $k=s^{-1}(hash+d*r)\%n$
- $k_{l}+k_{h}=s^{-1}(hash+d*r)\%n$
- $k_{l}=s^{-1}hash+s^{-1}dr-k_{h}\%n$

由上述可以构造如下一个格

$\begin{bmatrix} n & 0 & \dots & 0 & 0 & 0  \\0 & n & \dots& 0 & 0 & 0 \\ \vdots & \vdots & \ddots & \vdots & \vdots & \vdots  \\ 0 & 0 & \dots & n & 0  & 0 \\ A_{1}&A_{2}&\dots&A_{n}&B/n&0\\B_1&B_2&...&B_n& 0 &2^{l+1} \end{bmatrix}$

其中$A_i=r_i^{-1}s_{i},B_i=h*s_{i}^{-1}-k_{hi},B=2^{l+1}$

（一开始还想倒数第二位取出来 后来意识到这是群p上的运算,当然用k算d，就不需要考虑这个）

#### 代码实现

```python
#!/usr/bin/env sage
from Crypto.Util.number import long_to_bytes, bytes_to_long
import gmpy2
from sage.all import *
import numpy as np
r=[]
s=[]
k=[]
h=[]
u=[]
w=[]
d=100
a = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7FFFFFFC
b = 0x1C97BEFC54BD7A8B65ACF89F81D4D4ADC565FA45
p =0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7FFFFFFF
n =0x100000000000000000001f4c8f927aed3ca752257
E=EllipticCurve(GF(p), [a, b])
# G = E(0x4a96b5688ef573284664698968c38bb913cbfc82, 0x23a628553168947d59dcc912042351377ac5fb32)

#readline
with open("data.txt") as f:
    for i in range(d):
        res = f.readline()
        res=res.split(" ")
        r.append(int(res[0],16))
        s.append(int(res[1],16))
        k.append(int(res[2],16))
        h.append(int(res[3],16))
        u.append((int(res[0],16) * gmpy2.invert(int(res[1],16), n)))
        w.append(-QQ(int(res[3],16) * gmpy2.invert(int(res[1],16), n)-int(res[2],16)))

M=matrix(QQ,d+2,d+2)
B=2**64
for i in range(d):
    M[i,i]=QQ(n)
    M[100,i]=QQ(u[i])
    M[101,i]=(QQ(w[i]))
M[100,100]=QQ(B)/QQ(n)
M[101,101]=QQ(B)
# print(n)
# print(M)
m=M.LLL()
for i in range(102):
    tmp=m[i][-1]
    if tmp==B:
        print(i)
M=list(m)[1]
d=ZZ((M[-2])*n/B)
print(hex(n-d))
```

## DSA中的应用

### 2020高校战役

```python
import os
from random import SystemRandom
from string import ascii_letters, digits
from hashlib import sha256
from Crypto.Util import number
FLAG="XZITCTF{123456789123456789}"
random = SystemRandom()
def proof_of_work():
    s = os.urandom(10)
    digest = sha256(s).hexdigest()
    print(f"sha256(XXX + {s[3:].hex()}) == {digest}")
    try:
        x = input("Give me XXX in hex: ")
    except:
        print("Invalid input!")
        return False

    if len(x) != 6 or x != s[:3].hex():
        print("Wrong XXX!")
        return False
    return True
def genkey():
    # DSA
    q = number.getPrime(128)
    while True:
        t = random.getrandbits(1024-128-1)
        p = (t * 2*q + 1)
        if number.isPrime(p):
            break
    e = (p-1) // q
    g = pow(2, e, p)
    x = random.randint(1, q-1)
    y = pow(g, x, p)
    return {'y':y, 'g':g, 'p':p, 'q':q, 'x':x}
def sign(m, key):
    g, p, q, x = key['g'], key['p'], key['q'], key['x']
    k = random.randint(1, q-1)
    print(f"k.bit_length() == {k.bit_length()}")
    Hm = int.from_bytes(sha256(m.encode()).digest(), 'big')
    r = pow(g, k, p) % q
    s = (number.inverse(k, q) * (Hm + x*r)) % q
    return (r, s)
def verify(m, sig, key):
    r, s = sig
    y, g, p, q = key['y'], key['g'], key['p'], key['q']
    if not (0 < r < q) or not (0 < s < q):
        return False
    Hm = int.from_bytes(sha256(m.encode()).digest(), 'big')
    w = number.inverse(s, q)
    u1 = (w * Hm) % q
    u2 = (w * r) % q
    v = ( (pow(g, u1, p) * pow(y, u2, p)) % p ) % q
    return v == r
def _is_valid_name(name):
    if not (0 < len(name) <= 20):
        print("Invalid username length!")
        return False
    for c in name:
        if c not in ascii_letters + digits:
            print("Invalid character in username!")
            return False
    if name == "admin":
        print("Username can't be 'admin'")
        return False
    return True
def sign_up(key):
    name = input("Please input your username: ")
    if not _is_valid_name(name):
        return
    (r, s) = sign(name, key)
    sig = name.encode() + r.to_bytes(20, 'big') + s.to_bytes(20, 'big')
    print(f"Here is your signature in hex: {sig.hex().upper()}")
def sign_in(key):
    data = input("Please send me your signature: ")
    try:
        data = bytes.fromhex(data)
    except ValueError:
        print("Invalid signature format!")
        return
    if not (40 < len(data) <= 60):
        print("Invalid signature length!")
        return
    (name, r, s) = (data[:-40].decode(), data[-40:-20], data[-20:])
    sig = map(lambda x: int.from_bytes(x, 'big'), (r, s))
    if not verify(name, sig, key):
        print("Wrong signature!")
        return
    print(f"Welcome, {name}")
    if name == "admin":
        print(f"The flag is {FLAG}")
menu = '''
1. sign up
2. sign in
3. exit\
'''
def main():
    try:
        if not proof_of_work():
            return
        print("Generating DSA parameters...")
        key = genkey()
        print(f"p = {key['p']}")
        print(f"q = {key['q']}")
        print(f"g = {key['g']}")
        print(f"y = {key['y']}")
        while True:
            print(menu)
            choice = input("$ ")
            if choice == "1":
                sign_up(key)
            elif choice == "2":
                sign_in(key)
            elif choice == "3":
                print("Bye!")
                return
            else:
                print("Invalid choice!")
                continue
    except:
        pass
main()
```

#### 题目理解



#### 可行性分析

#### 代码实现

## LCG问题中的应用

### NPUCTF2020 babyLCG

```python
from Crypto.Util.number import *
from Crypto.Cipher import AES
from secret import flag

class LCG:
    def __init__(self, bit_length):
        m = getPrime(bit_length)
        a = getRandomRange(2, m)
        b = getRandomRange(2, m)
        seed = getRandomRange(2, m)
        self._key = {'a':a, 'b':b, 'm':m}
        self._state = seed
        
    def next(self):
        self._state = (self._key['a'] * self._state + self._key['b']) % self._key['m']
        return self._state
    
    def export_key(self):
        return self._key


def gen_lcg():
    rand_iter = LCG(128)
    key = rand_iter.export_key()
    f = open("key", "w")
    f.write(str(key))
    return rand_iter


def leak_data(rand_iter):
    f = open("old", "w")
    for i in range(20):
        f.write(str(rand_iter.next() >> 64) + "\n")
    return rand_iter


def encrypt(rand_iter):
    f = open("ct", "wb")
    key = rand_iter.next() >> 64
    key = (key << 64) + (rand_iter.next() >> 64)
    key = long_to_bytes(key).ljust(16, b'\x00')
    iv = long_to_bytes(rand_iter.next()).ljust(16, b'\x00')
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = flag + (16 - len(flag) % 16) * chr(16 - len(flag) % 16)
    ct = cipher.encrypt(pt.encode())
    f.write(ct)


def main():
    rand_iter = gen_lcg()
    rand_iter = leak_data(rand_iter)
    encrypt(rand_iter)


if __name__ == "__main__":
    main()
```

#### 题目理解



#### 可行性分析

#### 代码实现

## 其他

### 2023数字中国网络安全人才赛道

```python
from Crypto.Util.number import *
from secrets import flag
assert len(flag) == 38
t = 30
p = getPrime(512)
x = getPrime(512)
while x > p:
    x = getPrime(512)
rs = []
cs = []
ss = []
for i in range(t):
    r = getPrime(512)
    s = getPrime(400)
    c = (r * x + s) % p
    rs.append(r)
    cs.append(c)
    ss.append(s)

enc = pow(x,flag,p)
print(f'p = {p}')
print(f'rs = {rs}')
print(f'cs = {cs}')
print(f'enc = {enc}')
# p = 6897108443075981744484758716081045417854227543713106404294789655180105457499042179717447342593790180943415014044830872925165163457476209819356694244840079
# rs = [12844634549263053228759749264403637022740290008286987401585068952741935277415527678380021212624846722242500708422759563558995936977274580301379494195702461, 12251634003683452916928102291170339939586644029776192301741341674585154859358419625191986830852794085541953563738986709807899575511700135958334229151930861, 7051370666077542197248638013011793824477073777322219545882367881807130066168444134964571398112151848834032654978368255218649720738040945429837692857031957, 9773046862351952930368505593284546267554571295872377323111558071278701231472975791962979256551519533723988556870551885073742407630481198192389750289392107, 8883776497660138308720006912582738672888752344326928153810910221453595077711284302041512529457450211602787210761461172326429880594024187025419873043435877, 12056735137145460036580841038332100311160368843873164649606343042416896898793233249873902218683966283969721460087390120622254758027779960740926123005377571, 8819958747150954554494406068232243249186433676383469322817152210037563032056202909377825740775383087605647374150477096718956454225946093710691864988563109, 12246023449098354751049599873213988024512286270964608502444597112110163392131757813461977030270733012385926751192637938686124570227538910606279104888073013, 11308837998867241929817950595621831002334468993828126438599805989088017326675963100044309448653090403889186401929445861220402556074702741108929442867300279, 9184622887414209361516593101129556569811888214607556630094969763910426953786020755838094184972397480276666170685926425137063559394969166216392939257091541, 12896400069515890897430087815982545671830645201023665112429779640768899091287291452408369445919464144390726200808875066389240126909811239597092893733457339, 11227025698697471809912850435140886785315702278826761054472525227951791647003561270585720797267604996360933395122286757099101227901032364782594523739698877, 8162123490656317490361880020667919072708091053716891870691544217490126444997503404094174246087938828993696335191488583306443577208796794274099282013427247, 13366989889442670291461262313757977600095962057470863475519088648267301129719953368943419562144276679400967122727554764013132918505564677243979978807323041, 9920857455945408588203972193444437533164351309299040911469275059092031755811492460585653948481522995557801781838215407648572999358456612525812067538372579, 7139402473546047825312503780125417567716958846513076797328672521987900978293260385267945187604725349720103672258987935569856239987227455748213833342843243, 13108142660294572752252393081421368493392921884487755391460006730258159004638343897340537616297811742032405724656497443006056456690449881719305597286675631, 13276762958403786077380090195631980415297280849950287990717193547481553124160398455403123819234755237450529090601858784999113026218918277529515287668651121, 12463094640052886550696551772104539361264529587569204472038955376345085195998921095774583176899949596998985033050547755235409943131811058035802010421860899, 11307743131694864808301935844724645695851330736969875190167422024500753079857478680029193758960169072890576310607053767920339034290416580654771095674487943, 10053742503547378455068966704402695956702795408343604912294923217443553169726438945982031485796964462946592530592946335569560364464958066521486506177193131, 9703695763451799125258961776229325510814289358679213305418559381901496449849584244211834872313767844996255556721041007654625153809128987422992102292472533, 8148189465927721940294369879439913703690047528695196368949823197675716174991296513758196009346701553643721225250628151384047219921709201619262393792138023, 9114150910964237818418367840207724528917302406836157928223872622442928604249864486858755737149640683259834299165900696585038569188627682022002709058902291, 12273514376180781903469287345188404399033432117915094289694407562166649079228640510678711431664410226301556172582177240184695103942141430877677144285616059, 8355005721684425514882933910286584148305344580589623112959517428993968438533866906223777778058096962333203237111245328436600994120168924143849685728268811, 8957883838807471492147480816683526636019698464133185237668243268667169800811696770484487123560197988448434475112352005768286417529319182162245840523697001, 12168542584724814356632409768687396920143300559579648963851924568387314914334359305942685551210180448419674060219496395116081866784918059237133414041227833, 12285935007930825571672128346804313607196190465690759870758278705086034778808662886056460827935986285259185071514490942831585313190946386878622608868345563, 7719913817859572377164973343651155934060296607908537845256755472465025202751239980758950094865067751407889569369974011139801401586939119147773466111699913]
# cs = [3911325901261770731066343727353093385607196883601022244426857460074338420692610012414571623512152485474248169220030587839849722757773859682519433853455847, 4198555117325325874584019691418573071733167640213933749582347442518997588452211673143722179281773602455507001395983681009769848414007206268682184816168744, 4422173666634983234895098798813962037875417568235708524339826709271381748884936178371767574064794177416615710120223914725873239836121654705208614576413533, 3540260422555697887869627546208164711550015909378340105077652177481959576550678379723450981807556863572610759824660630418670546203733170058626755080797998, 6451498467498935201092514865627931677091078787997097414208430992183264950579022373372254486595458117887305393317663712337699331503725124287017134808484874, 3439629581963524351810430910737336124616316641656190641248434504621774235943514617301857917041111617104850245148746427180069743940612560718213177903427306, 4279468191481832212496939242093486044278976937965085475567008228061184947513156012369586970486543083130565628906296600553024574099481246534878242920637212, 4102135455518061133919027670571325279976222647984452353051395864554309521223498761823084717077102213648612826513661629599971609555235760152049549057234342, 329051927890365028889097463563711966673066795688728876214731188783168691555262156515161429328581094087585127929869064685419149676592073496155898360311360, 1674347209896897571502352451063188834938904430329951752111921115230349947823188121972980025563878887201507629419811736910690965020923751424101521816057970, 4779084317811375050159574994746297486592271247137823471375199626788956576998627181220489952507937768042501203098391966702297812537463211799837921684467541, 5240331815784322792144549873873658636726233093228415489098002982220769676718681132737794994708716389174162820721646744776624413735318240597745363490427584, 2689716894922604875455207695253665212853470308341743040957367957727155614199743562225147359614514189877983156892749669804800163252617480446565479990148021, 449708769594599088851244243076921016853502252396793496349534051273454215985560340288452398756880916680293627457774430655982228613348249480600180821975835, 1584603978331289335352997151059666773277943458357161051278658090420067023680231414255557805410288144092653121568766136372728095300982743309696347031121424, 4874260053151700374809337053763032489184725334196495160358275038586824027920238733886703163018450814805937363825223459277373073591021082276610135118976834, 3524374131362906900545297291947110177298862564718451821839794960169356082042548386553363480921097452902723033854749443288682983558847052843293666815425196, 6544123591499569232021913370293570477776709315008783531720886545784773471486769240711262562401683145937715612435213816372680189321141928790509490282629891, 4873861166228118967099569086478548167127431415017791678812419676791754466935832034870862000658789609084166891933970013849850146718379819943737269970654866, 4100817874436703071716655163972145036104985973164830547825929590871920825981241934633977227547934514142660786061291026657802357404024236287955309372489516, 343238276681348130286495167739162902430650061145485619903964358840996341335935043000395056684771452815629410388891486531126938900311458948803147120186532, 2683710724350412998770392318832434885304538325033159937379489319924346689197445720734209841902612235485016866254994045969716413020197296428323832404151182, 5909464641105704179999104311562416363090166762341644691188169716182958971270396007422581429813172933930581475771306034495224054972725230757675444731953480, 105593489999747649490909471306354863316673821363863362258853043970534652401274789197677558215188249074837829003335733211890211648501689656345824858507373, 4992379366542645691375959247465888889778118153982142100956809440855745659745235576280578316185469306620017845690312554043770651058126536040173113949524396, 6533456398244789907636779407045515567135195474284185379689518387558345997627435421582437390053234675991361808532278264077968540197407743744279106871716267, 5169360398767270275853790242315213671633880428212603766301308853363063092609582572957561138022806887895634140899640025570759919257615537375706008159680239, 203310740924699994885931266978520636166917734618272844754878785050509801614513144739164450834936178065792112797202959106365282699245578309060905297742706, 3143563289239398127009575193211845399079310618985464994769603542400451633289266080869317336163844517539211542909055869608349639432145332113320465388067087, 4016252180207572047405081190649590978593306403200098541033213590567723751195926093369984531729148621419589009515870336049849542537363832071754623330736088]
# enc = 1315637864146686255246675143589215932218700984880749264689270214639479160648747323586062096067740047809798944996253169402675772469028914904598116394230426

```

#### 题目理解

t表示轮数，一共有30轮对应的$    c = (r * x + s) \% p$题目中给出的是$p、rs、cs、以及enc$

按照题目目的，我们需要获得x,但是没有给出s，从题目表面上可以初步判断这是一个hnp隐藏数问题

#### 代码实现



### 西湖论剑2022 MyCurveErrorLearn

```python
from Crypto.Util.number import *
import os

flag = os.getenv('DASFLAG')

p = getPrime(1024)
a = getPrime(200)
b = getPrime(200)
E = EllipticCurve(GF(p), [a, b])
R = E.random_element()
P = E.random_element()

def XennyOracle(t):
    O = P + t*R
    return int(O[0]) - getPrime(163)


def task():
    for _ in range(30):
        op = int(input())
        if op == 1:
            XennyOracle(int(input()))
        elif op == 2:
            ss = int(input())

            if ss == P[0]:
                print('flag: ', flag)

try:
    task()
except Exception:
    print("Error. try again.")
```

### BCTF2018-guess number

```python
import random

from distlib.compat import raw_input
FLAG="xzitctf{123456789123456789}"
import gmpy2


def msb(k, x, p):
    delta = p >> (k + 1)
    ui = random.randint(x - delta, x + delta)
    return ui


def main():
    p = gmpy2.next_prime(2**160)
    for _ in range(5):
        alpha = random.randint(1, p - 1)
        # print(alpha)
        t = []
        u = []
        k = 10
        for i in range(22):
            t.append(random.randint(1, p - 1))
            u.append(msb(k, alpha * t[i] % p, p))
        print(str(t))
        print(str(u))
        guess = raw_input("Input your guess number: ")
        guess = int(guess)
        if guess != alpha:
            exit(0)


if __name__ == "__main__":
    main()
    print(FLAG)
```

#### 题目理解

https://kel.bz/post/hnp/这一题以及上面数字中国的那道题目其实都有点像这篇文章中的情形 按照这篇文章中的情形 作者给出一个独特的格子如下

![image-20230421101901518](https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230421101901518.png)

#### 代码实现



# 补充

## EHNP问题

![image-20230421102543129](https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230421102543129.png)

在文章中给出这样一个格子

![image-20230421113529512](https://gra1n-cn.oss-cn-hangzhou.aliyuncs.com/image-20230421113529512.png)

# 总结

我希望这篇文章的周期会长一点，也希望自己可以沉下心去做事情。目前这件事只当作一件忙碌之余的消遣，但希望可以让养成自己考虑问题尽可能充分的习惯。

# 参考

1. https://jsur.in/posts/2022-10-03-sekai-ctf-2022-ezmaze-writeup
2. https://zhuanlan.zhihu.com/p/581146119
3. https://crypto.sjtu.edu.cn/~hongrui/CIS2019/pnguyen.pdf
4. https://zhuanlan.zhihu.com/p/380148541
5. https://clingm.github.io/2022/11/18/Hidden-Number-Problem/
6. https://www.iacr.org/archive/crypto2009/56770333/56770333.pdf
7. https://cloud.tencent.com/developer/article/2242952
8. https://ctf-wiki.org/en/crypto/asymmetric/lattice/cvp/
9. https://sci-hub.ru/10.1007/978-3-0348-8295-8_23
10. https://www.anquanke.com/post/id/204846
11. An Introduction to Mathematical Cryptography (2014) - Hoffstein, Pipher, Silverman
12. The Dark Side of the Hidden Number Problem: Lattice Attacks on DSA
13. https://project-euphoria.dev/blog/29-seccon-2021/
14. https://kel.bz/post/hnp/
15. https://zhuanlan.zhihu.com/p/581335400
16. https://clingm.github.io/2022/11/22/Breaking-ECDSA-from-nonce-bits/
17. https://blog.csdn.net/forest_LL/article/details/124887974
18. https://www.youtube.com/watch?v=0jAAO8Xxj-I&t=24s
19. https://blog.msfakatsuki.com/Crypto/LLL-Lattice-deduction/
20. https://zhuanlan.zhihu.com/p/54483595
21. https://www.iacr.org/archive/crypto2009/56770333/56770333.pdf
22. https://eprint.iacr.org/2019/023.pdf
23. https://link.springer.com/chapter/10.1007/978-3-031-22301-3_8
24. https://link.springer.com/chapter/10.1007/978-3-540-74462-7_9
25. https://jsur.in/posts/2022-10-03-sekai-ctf-2022-ezmaze-writeup







