# K-Means

Different k-means initialization
* Random
    * Choose k (with no replacement) from data samples uniformly.
    * This can result in very close centers and very far centers which is bad for convergence.
* k-means++
    * Choose first center randomly from the data uniformly.
    * Choose the next cluster based on weighted probability. The weight can be obtained using the minimum distance squared of unselected points to all selected points. This way, the farther points will get more probability of being chosen.
    * Repeat above steps until k centers are selected.


# References
* [KMeans from scratch](https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670)
* [k-means++ paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
* [sklearn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
