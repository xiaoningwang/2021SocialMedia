# Pagerank and Katz centrality
###### 赵霖琳 2020213053032

### （一）中心性知识补充：

##### 1.简介：

​        在开始阐述PageRank和Katz中心性的概念之前，我们首先要了解什么是中心性 。中心性是用来度量结点在网络中的重要性的度量工具，对于网络中的单个结点或由多个结点组成的群体都可以定义中心性。对于单个结点的中心性中，主要分为度中心性（*degree centrality*）特征向量中心性（*eigenvector centrality*）Katz中心性，PageRank等，在这写中心度中，**后者是在前者的基础上进一步限制条件得到的**。所以在介绍Katz中心性和Pagerank之前，我们首先了解一下度中心性和特征向量中心性。

##### 1.1度中心性：

​		度（*Degree*）是刻画单个结点属性的最简单而又最重要的属性之一，度中心性主要针对有向图和无向图，在《社交媒体分析》的课程中我们也有过简单的了解。针对无向图（左图）结点*V<sub>i</sub>* 的度中心性为 $C_d(V_i)=\sum_{j=1}^nx_{ij}(i\neq j)$ 其中$j$表示与$i$相连接的结点，即为结点的度。针对有向图中心度既可以是出度（视为和群性) $C_d(V_i)=\sum_{j=1}^nd_{ij}^{out}$  也可以是入度（视为声望），也可以 $C_d(V_i)=\sum_{j=1}^nd_{ij}^{in}$ 是二者的和 $C_d(V_i)=\sum_{j=1}^nd_{ij}^{out}+d_{ij}^{in}$ ，例如我们可以根据一个人接受的关系（入度）来表示其受欢迎程度，根据他对外建立的关系（出度）来表示他的合群性。

​		但是仅仅用度来表示顶点的中心性是不严谨的，在规模不同的网络中，顶点连边的数量会受网络的规模影响，即规模越大的网络中 $C_d(V_i)$ 的值有可能越大。于是我们引入标准化概念，将上一步中得到的 $C_d(V_i)$ 除以其所连接的顶点个数:
$$
C_d^{'}(V_i)= \frac{C_d(V_i)}{n-1}
$$

###### (1-1)   

​ ![picture1](https://github.com/linlinzhao-moon/2021SocialMedia/blob/main/figure/zll_picture1.png)		![picture2](https://github.com/linlinzhao-moon/2021SocialMedia/blob/main/figure/zll_picture2.png)	

###### (1-2)

​			同时，如果我们给链接间赋予一个权重，那么我们可以得到一个邻接矩阵：

​	（1）加权无向图的邻接矩阵：

$$
\begin{pmatrix}
\ 0&1&0&1\\
\ 1&0&1&1\\
\ 0&1&0&0\\
\ 1&1&0&0\\
\end{pmatrix}
$$
###### (1-3)

​	（2）加权有向图的邻接矩阵：
$$
\begin{pmatrix}
\ 0&2&2&0\\4&0&0&0\\0&0&0&4\\1&1&1&0\\
\end{pmatrix}
$$

###### (1-4)

##### 1.2特征向量中心性：

​		特征向量中心性是整合了某个结点的邻居结点的中心性作为该结点的中心性，如果将这一理念放在日常生活的社交网络中，可以简单的抽象理解为——如果你认识的大牛越多，那么你也是一个大牛。这一概念的公式表达如下：
$$
C_e(v_i)= \frac{1}{\lambda}\sum_{j=0}^n\  A_{j,i}C_e(v_j)
$$
###### (1-5)

​		其中$${\lambda}$$是一个常量，$C_e(v_i)$是结点$v_i$的中心性。我们可以将上式写成矩阵的形式，特征向量中心性实际上就是对网络的邻接矩阵$A$进行特征分解，选择最大特征值所对应的特征向量作为各结点的中心性。即：
$$
\lambda C_e=A^TC_e
$$

###### (1-6)

​		其中，$C_e$是邻接矩阵$A^T$的特征向量，$\lambda$是对应的特征值。但是由于中心性要求大于零，于是引入了非负矩阵中的佩龙-弗罗宾尼斯定理*(Perro-frobenius theorem)*：假设$A\in{R^{n\times n}}$是强连通图的邻接矩阵，或者$A$：$A_{i,j}>0$（即$A$是一个正的$n\times n$矩阵），存在一个正实数$\lambda_{max}$满足$\lambda_{max}$是矩阵$A$的特征值，并且$A$的其余特征值均严格小于$\lambda_{max}$，那么$\lambda_{max}$所对应的特征向量$V=(v_1,v_2,…,v_n)$满足$\forall v_i>0$。这样特征向量$V$就可以用来描述各结点的中心性了。

​		**例如，**以（1-2）中左图为例，引入一个$4\times1$的向量$X^T=(1,2,3,4)$，向量的值对应图中的4个顶点1-4，我们利用(1-5)式来计算每个点的中心度：
$$
\begin{pmatrix}0&1&0&1\\1&0&1&1\\0&1&0&0\\1&1&0&0\end{pmatrix}\begin{pmatrix}1\\2\\3\\4\end{pmatrix}=\begin{pmatrix}6\\8\\2\\3\end{pmatrix}
$$

###### (1-7)

​		设得到的结果向量$B^T=(6,8,2,3)$就是结点1-4的一个大概的中心性，它用邻接矩阵的第$i$行与$X^T$的第一列相乘，去获每个和第$i$个顶点相连的点的值，将他们加起来后得到第$i$个顶点的中心性。

​		总的来说，特征向量的中心性是在度中心性的基础上的自然拓展，而Kazt中心性也是在特征向量的中心性上改进完善的。在简单的了解度中心性和特征向量中心性后，我们将来具体介绍Kazt中心性和Google有名的PageRank。



### （二）Katz中心性

##### 1.简介：

​		katz中心性是在特征向量中心性的基础上限制条件而进一步得到的。举个例子简单来说，具有许多连接的网站，即使它没有被其他网站连接（不在强链接组件中）它依旧可以被认为是重要的，例如引文网络。像上述的例子，由于入度为0，故而其特征向量的中心性为0。为了解决这个问题，我们在特征向量的基础上加入了一个偏差项$\beta$：
$$
C_{Katz}(v_i)=\alpha\sum_{j=0}^nA_{j,i}C_{Katz}(V_j)+\beta
$$

###### (2-1)

​		其中 $\alpha$ 和 $\beta$ 是常数项，上式中第一项是特征向量中心性项，可以参考 (1-5) 是顶点 $i$ 的特征向量中心性。在加上偏差项后，即使网络中某个顶点 $i$ 的入度为零，其Katz中心性都不会等于零，这样 $i$ 指向其他顶点时，这些顶点的中心性也会进一步提升。可将上述写成矩阵的形式，即：
$$
C_{Katz}=\alpha A^TC_{Katz}+\beta1
$$

###### (2-3)

​		为解得Katz中心性，移项可得：
$$
C_{Katz}=(I-\alpha A)^{-1}\beta
$$

###### (2-3)

​		为了方便计算，我们通常将 $\beta$ 的值设为 $1$：
$$
C_{Katz}=(I-\alpha A)^{-1}\cdot1
$$

###### (2-4)		

​		Katz中心性除了引入偏差项 $\beta$ 解决入度为零的顶点的特征向量中心性为零外，另一个重要的方面时引入了一个自由参数 $\alpha$ 也可成为衰减因子，其取值通常在 $(0,1)$ 之间，目的是为一对结点间的每条路径分配一个权重以控制特征向量和之间的平衡。**以(1-2)左图为例**，我们要计算顶点1的中心性取 $\alpha =0.5$ 那么与1直接相连的2，4的权重均为 $(0.5)^1=0.5$ ，而3通过2与1相连，故而3的权重为 $(0.5)^2=0.25$。

​		在参数 $\alpha$ 的选择上也有着一定的方法，首先 $ \alpha$ 的值不能任意大。如果我们让 $\alpha \rightarrow 0$，那么在（2-1）式中只有常数项存在，并且所有顶点都有相同的中心性 $\beta$，以（2-1）式为例，当我们从0增加 $\alpha$ 时，顶点 $i$ 的中心性增加最终达到他们发散的点，此时的条件为：
$$
det(A-\alpha ^{-1}I)\cdot1=0
$$

###### (2-5)

​		上式只是一个特征方程，其根 $shi$ 等于邻接矩阵的特征值（假设为 $k$ ），换而言之，当 $\alpha$ 慢慢增加到 $\frac{1}{k}$ 时上式为零顶点的中心性达到它的发散点，但是我们都希望中心性收敛，所以 $\alpha$ 的取值应在$(0,\frac{1}{k})$ 之间。对于无向网络，其最大特征值是1，所以一般也说 $\alpha$ 的值取 $(0,1)$。



##### 2.代码实现：

```
def katz_centrality(G, alpha=0.1, beta=1.0,  #构造计算Katz中心性的函数
					max_iter=1000, tol=1.0e-6,
					nstart=None, normalized=True,
					weight = 'weight'):
	
	from math import sqrt

	if len(G) == 0:
		return {}

	nnodes = G.number_of_nodes()

	if nstart is None:

		# 选择入度为零的当起始向量
		x = dict([(n,0) for n in G])
	else:
		x = nstart

	try:
		b = dict.fromkeys(G,float(beta))
	except (TypeError,ValueError,AttributeError):
		b = beta
		if set(beta) != set(G):
			raise nx.NetworkXError('beta dictionary '
								'must have a value for every node')

	# 迭代次数：
	for i in range(max_iter):
		xlast = x
		x = dict.fromkeys(xlast, 0)

		# do the multiplication y^T = Alpha * x^T A - Beta
		for n in x:
			for nbr in G[n]:
				x[nbr] += xlast[n] * G[n][nbr].get(weight, 1)
		for n in x:
			x[n] = alpha*x[n] + b[n]

		# 检查收敛性
		err = sum([abs(x[n]-xlast[n]) for n in x])
		if err < nnodes*tol:
			if normalized:

				# 规范化向量
				try:
					s = 1.0/sqrt(sum(v**2 for v in x.values()))
					
				except ZeroDivisionError:
					s = 1.0
			else:
				s = 1
			for n in x:
				x[n] *= s
			return x

	raise nx.NetworkXError('Power iteration failed to converge in '
						'%d iterations.' % max_iter)

```

```
>>> import networkx as nx ##引用了networkx包构建网络
>>> import math
>>> G = nx.path_graph(4)
>>> phi = (1+math.sqrt(5))/2.0 # largest eigenvalue of adj matrix
>>> centrality = nx.katz_centrality(G,1/phi-0.01)
>>> for n,c in sorted(centrality.items()):
... print("%d %0.2f"%(n,c))
```

```
0 0.37 ##0-3各顶点的Katz中心性值
1 0.60
2 0.60
3 0.37
```



### （三）PageRank

##### 1.简介：

​		Katz中心性虽然解决了特征向量中心性中入度为零的定点中心性的问题，但仍存在不适用的场合。我们知道一个具有高Katz中心性的顶点指向许多其他顶点，那么这些顶点也具有较高的中心性。但如果一个高Katz中心性的顶点指向100万个其他顶点，这会使得这100万个被指向的顶点都具有高中心性，但这有时是不恰当的。许多情况下，如果一个顶点只是被指向的众多定点中的一个，其从高Katz中心性顶点连接的一条边所得到的中心性由于与许多其他顶点共享而被稀释，所以这个顶点的中心性并没有很高。

​		以生活中的社交网络举例，如果校长认识同学A同时也认识副校长，即校长与副校长和同学A都有连接，校长的中心性显然是很高的，但与此同时能说明同学A和副校长的是同等重要的吗？这显然是不合理的，因为同学A只是校长认识的同学ABCDEF中的一个，他的中心地位会被稀释。

​		所以在计算上我们要对Katz中心性的计算有所改进，不能直接对相邻顶点的中心性求和，因为存在稀释，所以我们可以考虑用相邻顶点的中心性和该顶点的出度求商：
$$
C_p(v_i)=\alpha\sum_{j=0}^nA_{j,i}\frac{C_p(V_j)}{d_j^{out
}}+\beta
$$

###### (3-1)

​		也可将其写成矩阵的形式：
$$
C_p=\alpha A^TD^{-1}C_p+\beta \cdot 1
$$

###### (3-2)

​		其中，$1$ 是向量 $(1,1,1……)$，$D$ 是一个对角矩阵，$D_{ii}=max\lbrace d_j^{out},1\rbrace$ 为求得PageRank中心性的值，像katz中心性一样通常我们设 $\beta=1$，移项可得：
$$
C_p=\beta (I-\alpha A^TD^{-1})^{-1}\cdot 1=(I-\alpha AD^{-1})\cdot1=D(D-\alpha)^{-1}\cdot1
$$

###### (3-3)

​		与Katz中心性的公式一样，PageRank中也包含了一个自由参数 $\alpha$ ，在上一部分的（2-5）中我们有具体介绍，这里和上一部分一样，一般选择 $\alpha \in (0,1) $。



##### 2.代码实现：

​		PageRank作为Google网页排名的核心算法，其算法也是机器学习的经典算法，以下将用PageRank去挖掘希拉里邮件中的重要任务关系。

```
import pandas as pd

import networkx as nx

import numpy as np

from collections import defaultdict

import matplotlib.pyplot as plt

emails = pd.read_csv("./input/Emails.csv")	# 数据加载

file = pd.read_csv("./input/Aliases.csv") 	# 读取别名文件

aliases = {}

for index, row in file.iterrows():

    aliases[row['Alias']] = row['PersonId']

file = pd.read_csv("./input/Persons.csv")	# 读取人名文件

persons = {}

for index, row in file.iterrows():

    persons[row['Id']] = row['Name']

def unify_name(name):			# 针对别名进行转换

    name = str(name).lower()	# 姓名统一小写

    name = name.replace(",","").split("@")[0]	 # 去掉, 和 @后面的内容

    if name in aliases.keys():	# 别名转换

        return persons[aliases[name]]

    return name

def show_graph(graph, layout='spring_layout'):	# 画网络图

    if layout == 'circular_layout':	#使用Spring Layout布局，类似中心放射状

        positions=nx.circular_layout(graph)

    else:

        positions=nx.spring_layout(graph)

    	# 设置网络图中的节点大小，大小与 pagerank 值相关，因为 pagerank 值很小所			# 以需要*20000

    nodesize = [x['pagerank']*20000 for v,x in graph.nodes(data=True)]

    # 设置网络图中的边长度

    edgesize = [np.sqrt(e[2]['weight']) for e in graph.edges(data=True)]

    # 绘制节点

    nx.draw_networkx_nodes(graph, positions, node_size=nodesize, alpha=0.4)

    # 绘制边

    nx.draw_networkx_edges(graph, positions, edge_size=edgesize, alpha=0.2)

    # 绘制节点的 label

    nx.draw_networkx_labels(graph, positions, font_size=10)

    # 输出希拉里邮件中的所有人物关系图

    plt.show()

# 将寄件人和收件人的姓名进行规范化

emails.MetadataFrom = emails.MetadataFrom.apply(unify_name)

emails.MetadataTo = emails.MetadataTo.apply(unify_name)

# 设置遍的权重等于发邮件的次数

edges_weights_temp = defaultdict(list)

for row in zip(emails.MetadataFrom, emails.MetadataTo, emails.RawText):

    temp = (row[0], row[1])

    if temp not in edges_weights_temp:

        edges_weights_temp[temp] = 1

    else:

        edges_weights_temp[temp] = edges_weights_temp[temp] + 1

# 转化格式 (from, to), weight => from, to, weight

edges_weights = [(key[0], key[1], val) for key, val in edges_weights_temp.items()]

# 创建一个有向图

graph = nx.DiGraph()

# 设置有向图中的路径及权重 (from, to, weight)

graph.add_weighted_edges_from(edges_weights)

# 计算每个节点（人）的 PR 值，并作为节点的 pagerank 属性

pagerank = nx.pagerank(graph)

# 将 pagerank 数值作为节点的属性

nx.set_node_attributes(graph, name = 'pagerank', values=pagerank)

# 画网络图

show_graph(graph)

# 将完整的图谱进行精简
# 设置 PR 值的阈值，筛选大于阈值的重要核心节点

pagerank_threshold = 0.005

# 复制一份计算好的网络图

small_graph = graph.copy()

# 剪掉 PR 值小于 pagerank_threshold 的节点

for n, p_rank in graph.nodes(data=True):

    if p_rank['pagerank'] < pagerank_threshold:

        small_graph.remove_node(n)

# 画网络图, 采用 circular_layout 布局让筛选出来的点组成一个圆

show_graph(small_graph, 'circular_layout')
```

运行结果如下：

![picture3](https://github.com/linlinzhao-moon/2021SocialMedia/blob/main/figure/zll_picture3.png)

![picture4](https://github.com/linlinzhao-moon/2021SocialMedia/blob/main/figure/zll_picture4.png)

### (四)参考资料：

[1] Networks: An Introduction

[2] 汪小帆，李翔，陈关荣，《网络科学导论》。

[3] [Katz Centrality (Centrality Measure)]([Katz Centrality (Centrality Measure) - GeeksforGeeks](https://www.geeksforgeeks.org/katz-centrality-centrality-measure/))

[4] [PageRank算法]([(19条消息) PageRank算法_黄规速博客:学如逆水行舟，不进则退-CSDN博客_pagerank](https://blog.csdn.net/hguisu/article/details/7996185))



