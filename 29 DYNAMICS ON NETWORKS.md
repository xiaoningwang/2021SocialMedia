网络动力学 DYNAMICS ON NETWORKS
于若琳 2020213053003
（一）简介
网络动力学是指动力学模型在不同网络上的性质与相应网络的静态统计性质的联系,关于网络系统的由网络流量作为状态量的动力学。
网络动力学性质的基本研究对象包括已知和未知的静态几何量。
如果我们发现了某个模型在某一网络上有某种特殊的表现,那么可以认为是这一网络的某种特征影响了这个模型的表现。
这种特征有可能是已经得到研究的这种网络的几何特性,也有可能是没有被发现的几何特征，那么前者将印证网络上这些几何量的重要性,而后者将会推动网络本身研究的发展。

（二）原理分析
现在让我们将上一节的一些想法应用于网络上的动态系统。
首先，我们需要明确这些系统的含义。通常，我们的意思是我们有独立动态变量 xi，yi,
在我们网络的每个顶点 i 上，它们是仅沿网络的边缘耦合在一起。
也就是说，当我们写出我们的方程时,变量 xi 的时间演化，出现在该等式中的各个项仅涉及 xi顶点 i 上的其他变量，或网络中与 i 相邻的顶点上的一个或多个变量。 
那里是没有涉及非相邻顶点上的变量的项，也没有涉及非相邻顶点上的变量的项多个相邻顶点。
这种类型的动力系统的一个例子是我们的概率方程感染 SI 流行模型的网络版本中的一个顶点。


<center>$\FRAC{dx_{i}}{dt}=\beta(1-x_{i})\sumA_{ij}x_{j}$</center>

这个方程只有涉及由边连接的变量对的项，因为这些是唯一Aij非零的对。
对于每个顶点只有一个变量的系统，我们可以写出一个一般的一阶方程


<center>$\FRAC{dx_{i}}{dt}=f_{i}(x_{i})+\sumA_{ij}g_{ij}(x_{i},x_{j}),

我们将涉及相邻顶点变量的项与涉及相邻顶点变量的项分开,ƒi作为指定一个顶点的内在动力学——它指定了如何变量 xi将在顶点之间没有任何连接的情况下演化，即如果所有i的Aij = 0,相反，gij 描述了连接本身的贡献； 它代表了不同顶点上的变量之间的耦合。
请注意，我们指定了不同的函数 ƒi和gij 对于每个顶点或顶点对，所以每个顶点遵循的动力学可能不同。 然而，在许多情况下，当每个顶点代表类似的东西——比如流行病模型中的一个人——每个顶点的动态可能是相同的，或者至少足够相似以至于我们可以忽略任何
差异。在这种情况下，方程中的函数对所有顶点都相同，并且方程变成

<center>$\FRAC{dx_{i}}{dt}=f(x_{i})+\sumA_{ij}g_{ij}(x_{i},x_{j}),

我们将假设情况就是如此。 我们还将假设网络是无向的，所以 Aij 是对称的——如果 xi
受 xj 影响那么xj同样受到影响（但是请注意，我们不假设函数g在其参数中是对称的：g(u, v)≠ g(v, u).) 同样，方程的SI模型。


(三)模型分析
让我们尝试将线性稳定性分析工具应用于方程。假设我们能够找到方程的一个不动点。 通过求解联立方程。在这种情况下找到固定点意味着为每个顶点找到一个值i——不动点是完备集。另请注意，通常固定点的位置取决于网络上发生的特定动态过程（通过函数 ƒg) 以及网络结构（通过邻接矩阵）。如果其中任何一个发生变化，则固定点的位置也会发生变化。现在我们可以通过编写、执行
同时在所有变量中进行多重泰勒展开式，并删除二阶项少量及更高。

我们立即看到，如果所有特征值 μr 的实部是负数，那么 cr(t)——和，因此，对于所有 r，∈—随时间衰减，我们的不动点将是吸引的。如果真实的部分都是正固定点将被排斥。如果有些是积极的，有些是消极的，那么固定点是一个马鞍，虽然，和以前一样，这也许最好看作是一种排斥固定点：靠近鞍座的流动至少有一个排斥方向，这意味着一个在这样一个点附近启动的系统一般不会停留在它附近，无论是否其他方向是否吸引。

（四）应用实例
假设你收到一封邮件，主题行包含“check this out”，这通常是垃圾邮件包含的短语。单凭这一信息，而没有检查发件人或邮件内容，这封邮件是垃圾邮件的概率有多大？
Solution：
这是一个条件概率的问题，是在寻求以下表达式的值：
P[spam | "check this out"]

此外，假设收到的所有邮件中40%是垃圾邮件，其余60%是你想要接收的邮件。此外，假设1%的垃圾邮件在主题行中包含短语“check this out"，有0.4%的非垃圾邮件也在主题行中包含这个短语。因此，可以得到以下概率值：

P[spam]=0.4
P["check this out"|spam]=0.01
P["check this out"|not spam]=0.004
运用贝叶斯规则，可得：
P[spam|"check this out"]=$\frac{P["chenck this out"|spam]*P[spam]}{P["check this out"]}

因此，只需要计算P ["check this out" ]的概率值即可得所求；即需要计算短语"check this out在所有邮件中的发生概率，有：

P["check this out"]=P[spam]∗P["check this out"|spam]+P[notspam]∗P["check this out"|not spam]=0.4×0.01+0.6×0.004=0.064

带入以上数值，得到：

P[spam|"check this out"]=$\frac{0.004}{0.0064}=0.625

上述分析和计算结果说明，当采用了短语"check this out"作为鉴别信息时，识别这封邮件为垃圾邮件的概率将高于原始概率统计0.4.
因此，我们可以将主题行中出现这个短语看成是一个弱信号，为我们提供该邮件是否为垃圾邮件的证据。
在实践中，检测每一封邮件多种不同的信号，包括邮件的正文、邮件的主题、发送者的属性（你是否认识他们？他们的邮件地址有什么特点？）、邮件服务的属性，以及一些其他的特征。


参考文献
1.《Networks An introduction》M.E.J Newman

2.《网络动力学模型之信息级联》