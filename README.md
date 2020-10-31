# 近似最近邻搜索的论文、阅读笔记与实现分享

这里是一些近似最近邻搜索相关的论文、我们阅读过程中产生的[笔记](./notes)，分享时制作的[Slides](./slides)，以及论文中算法的简单实现。
在这里记录和分享，希望能为对近似近邻搜索感兴趣的同学带来帮助。
所有资料均来自于互联网，如有侵权，请联系我们。

目录
---
1. [[PQ]](./papers/Product_Quantization_for_Nearest_Neighbor_Search.pdf)
    **Product Quantization for Nearest Neighbor Search** (Jégou & al. TPAMI 2011.).

    Notes: [pq.pdf](./notes/PQ.pdf), [pq.md](./notes/PQ.md).

    Slides: [yx_zyh_0826](slides/PQ_sharing_yx_bosszou_20200826.pdf), [cqx_yk_0826](slides/PQ_sharing_cqx_yk_20200826.pdf).

2. [[PSTree]](./papers/Product_Split_Trees.pdf)
    **Product Split Trees** (Babenko & Lempitsky. CVPR 2017.).

3. [[IMI]](./papers/The_Inverted_Multi-Index.pdf)
    **The Inverted Multi-index** (Babenko & Lempitsky. TPAMI 2015.).

4. [[LOPQ]](./papers/Locally_Optimized_Product_Quantization_for_Approximate_Nearest_Neighbor_Search.pdf)
    **Locally Optimized Product Quantization for Approximate Nearest Neighbor Search** (Kalantidis & Avrithis. CVPR 2014.).

5. [[AQ]](./papers/Additive_Quantization_for_Extreme_Vector_Compression.pdf)
    **Additive Quantization for Extreme Vector Compression** (Babenko & Lempitsky. CVPR 2014.).

6. [[NO_IMI]](./papers/Efficient_Indexing_of_Billion-Scale_datasets_of_deep_descriptors.pdf)
    **Efficient Indexing of Billion-scale Datasets of Deep Descriptors** (Babenko & Lempitsky. CVPR 2016.).

    Notes:

    Slides: [cqx_yk_0912](slides/NO_IMI_sharing_cqx_yk_20200912.pdf).

7. [[TQ]](./papers/Tree_Quantization_for_Large-Scale_Similarity_Search_and_Classification.pdf)
    **Tree Quantization for Large-scale Similarity Search and Classification** (Babenko & Lempitsky. CVPR 2015.).

8. [[OPQ]](./papers/Optimized_Product_Quantization.pdf)
    **Optimized Product Quantization** (Ge, He, Ke & Sun. TPAMI 2013).

9. [[RE_IVF]](./papers/Revisiting_the_Inverted_Indices_for_Billion-Scale_Approximate_Nearest_Neighbors.pdf)
    **Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors** (Baranchuk, Dmitry, Babenko & Malkov. ECCV 2018).

    Notes:

    Slides: [cqx_yk_0926](/slides/Revisiting_inverted_index_sharing_cqx_yk_20200926.pdf).

10. [[HNSW]](./papers/Efficient_and_robust_approximate_nearest_neighbor_search_using_Hierarchical_Navigable_small_world_graphs.pdf.pdf)
    **Efficient and robust approximate nearest neighbor search using Hierarchicalr Navigable small world graphs** (Malkov & al. TPAMI 2016.)



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
