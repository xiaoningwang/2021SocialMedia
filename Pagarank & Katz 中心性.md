# PageRank & Katz 中心

- ### 中心性

​        在开始阐述PageRank和Katz中心性的概念之前，我们首先要了解什么是中心性 。中心性是用来度量结点在网络中的重要性的度量工具，对于网络中的单个结点或由多个结点组成的群体都可以定义中心性。对于单个结点的中心性中，主要分为度中心性*（degree centrality）*特征向量中心性*（eigenvector centrality）* Katz中心性，PageRank等，在这写中心度中，后者是在前者的基础上进一步限制条件得到的。所以在介绍Katz中心性和Pagerank之前，我们首先了解一下度中心性和特征向量中心性。

##### 度中心性：

​		度*（Degree）*是刻画单个结点属性的最简单而又最重要的属性之一，度中心性主要针对有向图和无向图，在《社交媒体分析》的课程中我们也有过简单的了解。针对无向图（左图）结点*V<sub>i</sub>*的度中心性为*C<sub>d</sub>(V<sub>i</sub>)=d<sub>i</sub>* ，即为结点的度。针对有向图中心度既可以是出度（视为和群性）*C<sub>d</sub><sup>out</sup>(V<sub>i</sub>)=d<sub>i</sub><sup>out</sup>*也可以是入度（视为声望）*C<sub>d</sub><sup>in</sup>(V<sub>i</sub>)=d<sub>i</sub><sup>in</sup>*，也可以是二者的和*C<sub>d</sub>(V<sub>i</sub>)=d<sub>i</sub><sup>in</sup>+d<sub>i</sub><sup>out</sup>* 

<img src="D:\QQ图片20211019214425.png" alt="图片" style="zoom:40%;" />                  <img src="D:\QQ图片20211019214152.png" alt="图片" style="zoom:40%;" />

​			同时，如果我们给链接间赋予一个权重，那么我们可以得到一个邻接矩阵：

​	（1）加权无向图的邻接矩阵：

​		
$$
\begin{pmatrix}
\ 0&1&0&1\\
\ 1&0&1&1\\
\ 1&0&0&0\\
\ 1&1&0&0\\
\end{pmatrix}
$$
​	（2）加权有向图的邻接矩阵：
$$
\begin{pmatrix}
\ 0&2&2&0\\4&0&0&0\\0&0&0&4\\1&1&1&0\\
\end{pmatrix}
$$


##### 特征向量中心性：

​		特征向量中心性是整合了某个结点的邻居结点的中心性作为该结点的中心性：
$$
C_e(v_i)= \frac{1}{\lambda}\sum_{j=0}^n\  A_{j,i}C_e(v_j)
$$
其中$${\lambda}$$是一个常量，$C_e(v_i)$是结点$v_i$的中心性。我们可以将上式写成矩阵的形式，特征向量中心性实际上就是对网络的邻接矩阵$A$进行特征分解，选择最大特征值所对应的特征向量作为各结点的中心性。即：
$$
\lambda C_e=A^TC_e
$$
其中，$C_e$是邻接矩阵$A^T$的特征向量，$\lambda$是对应的特征值。但是由于中心性要求大于零，于是引入了非负矩阵中的佩龙-弗罗宾尼斯定理*(Perro-frobenius theorem)*：假设$A\in{R^{n\times n}}$是强连通图的邻接矩阵，或者$A$：$A_{i,j}>0$（即$A$是一个正的$n\times n$矩阵），存在一个正实数$\lambda_{max}$满足$\lambda_{max}$是矩阵$A$的特征值，并且$A$的其余特征值均严格小于$\lambda_{max}$，那么$\lambda_{max}$所对应的特征向量$V=(v_1,v_2,…,v_n)$满足$\forall v_i>0$。这样特征向量$V$就可以描述各结点的中心性了。

​		在简单的了解



- ### Katz中心性

  ​		



- ### PageRank