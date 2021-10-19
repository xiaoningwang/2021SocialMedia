# PageRank & Katz 中心性

- ### 中心性

  ​		在开始阐述PageRank和Katz中心性的概念之前，我们要知道前两者都是单个结点中心性的分支之一。中心性*（centrality）*是用来度量结点在网络中的重要性，对于网络中的单个结点或由多个结点组成的群体都可以定义中心性。对于单个结点的中心性中，主要分为度中心性*（degree centrality）*特征向量中心性*（eigenvector centrality）*Katz中心性，PageRank，中间中心性和接近中心性。

- ### Katz中心性

  ​		在关于网络的大量研究中都致力于中心性*(centrality)*的概念，可能对于重要性这一说法有不同的定义，但看起来最简单的中心度度量就是一个顶点的度数。在有向网络中，顶点既有出度也有入度，它们都可以用来衡量顶点的中心地位。在*Networks an Introduction*这一本书的研究中解决了这个问题——“网络中最重要或最中心的顶点是什么？”
  
  ​		虽然度中心性是一个简单的中心性度量的概念，但它很有启发性。例如：
  
  >  it seems reasonable to suppose that individuals who have connections to
  > many others might have more influence, more access to information, or more prestige than those who have fewer connections.
  
  >  A non-social network example is the use of citation counts in the evaluation of scientific papers. The number of citations a paper receives from other papers, which is simply its in-degree in the citation network, gives a crude measure of whether the paper has been influential or not and is widely used as a metric for judging the impact of scientific research.
  
  ​			