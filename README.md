# 近似最近邻搜索的论文、阅读笔记与实现分享

这里是一些近似最近邻搜索相关的论文、我们阅读过程中产生的笔记，以及论文中算法的简单实现。
在这里记录和分享，希望能为对近似近邻搜索感兴趣的同学带来帮助。
所有资料均来自于互联网，如有侵权，请联系我们。

目录
---
- [[PQ] Product Quantization for Nearest Neighbor Search (Jégou & al. TPAMI 2011.)](./papers/Product_Quantization_for_Nearest_Neighbor_Search.pdf)
- [[PQ] Product Split Trees (Babenko & Lempitsky. CVPR 2017.)](./papers/Product_Split_Trees.pdf)
- [[IVF] The Inverted Multi-index (Babenko & Lempitsky. TPAMI 2015.)](./papers/The_Inverted_Multi-Index.pdf)

## 什么是近似最近邻搜索？

### 最近邻搜索

最近邻搜索问题（Nearest Neighbor, NN）是指根据数据之间的相似性，在数据库中找到和目标数据最相似的点的问题。

比如说图像识别领域，给定一张图片，如何在图片集中找到和它最相似的图像；在电商推荐推荐系统中，给定一个用户喜欢的商品，如何在所有商品中找到和这个商品类似的一些商品进行推荐；在模式识别领域的车牌识别中，如何将车牌照片上的数字字母识别成正确的数字字母；写电子文档时进行的拼写检查、语法检查，如何知道文档中的单词、词组是否正确等等等等，这些都是最近邻搜索的应用场景。

图片、商品等数据在数据库中一般用多维特征向量（vector）表示，数据之间的相似性则转换为它们的特征向量在向量空间之间的距离。衡量向量之间的距离的方式有很多，比如欧式距离，内积相似度和 Jaccard 相似度等等。

> 目前有些数据的特征向量是由 embedding 技术训练出来的，因此特征向量也会被称作 embeddings

最近邻搜索是全量搜索，假设向量的维度是 *D*，数量是 *N*，那么最近邻搜索的时间复杂度是 *O(DN)*，也就是说当搜索数据集特别大，向量维度特别高的时候，最近邻搜索计算量就变得非常大，难以应用在实际问题。所以近似最近邻搜索广受关注。

### 近似最近邻搜索

近似最近邻搜索（Approximate Nearest Neighbor, ANN）顾名思义，搜索出来的结果不要求是精确结果，只需要和精确结果相近即可。ANN的核心思想是牺牲精度来换取速度。

ANN 的方法主要分为两类，一类是基于哈希的方法，第二类是基于量化（Quantization）的方法，我们阅读的论文主要是和量化相关的内容。