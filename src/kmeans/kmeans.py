import numpy as np


class KMeans:
    def __init__(
        self,
        n_clusters: int,
        init: str,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.fitted = False

    def _kmeans_plusplus(self, x):
        # step 1a: select random first centroid
        selected_idx = [np.random.choice(len(x))]
        for i in range(1, self.n_clusters):
            # step 1b: choose next centroid with random choice with weighted probs based
            # on squared distance from selected vs unselected
            unselected_idx = np.setdiff1d(np.arange(len(x)), selected_idx)
            centers = x[selected_idx]  # (i, f)
            unselected = x[unselected_idx]  # (b-i, f)
            # select min distance from selectec to all unselected, (b-i,)
            # this is kinda similar to sdf in computer graphics
            min_dist_sq = ((unselected[:, None, :] - centers) ** 2).sum(-1).min(-1)
            # random choice based on normalized min_dist_sq weighting
            min_dist_sq = min_dist_sq / min_dist_sq.sum()
            selected_idx.append(np.random.choice(unselected_idx, p=min_dist_sq))
            # step: 1c: repeat the loop
        return x[selected_idx]

    def _init_cluster_centers(self, x: np.ndarray):
        """Create the first clusters based on init"""
        if self.n_clusters > len(x):
            raise ValueError("`n_clusters` must be <= num samples")
        # x is (batch, feat) (b, f)
        if self.init == "random":
            chosen = np.random.choice(len(x), self.n_clusters, replace=False)
            return x[chosen]
        elif self.init == "k-means++":
            return self._kmeans_plusplus(x)
        else:
            raise ValueError("init must be either `random` or `k-means++`")

    def _kmeans_loop(self, x, centers) -> tuple:
        """A single kmeans main loop, returns final centers and inertia"""
        # x is (b, f)
        # centers is (n_clusters, f)
        last_centers = centers.copy()
        for i in range(self.max_iter):
            # dist_sq is (b, n_clusters)
            # HACK distance squared can still work, no need for sqrt
            dist_sq = ((x[:, None, :] - centers) ** 2).sum(-1)
            closest_idx = dist_sq.argmin(-1)  # (b,)
            for cluster_idx in range(self.n_clusters):
                # update cluster center according to the same idx mean
                members = x[closest_idx == cluster_idx]  # (_, f)
                if len(members) == 0:
                    continue
                centers[cluster_idx] = members.mean(0)  # (f,)
            # frobenius norm
            if np.linalg.norm(last_centers - centers) <= self.tol:
                last_centers = centers.copy()
                break
            last_centers = centers.copy()
        # calculate inertia
        inertia = 0.0
        dist_sq = ((x[:, None, :] - last_centers) ** 2).sum(-1)
        closest_idx = dist_sq.argmin(-1)  # (b,)
        for cluster_idx in range(self.n_clusters):
            # update cluster center according to the same idx mean
            members = x[closest_idx == cluster_idx]  # (_, f)
            if len(members) == 0:
                continue
            inertia += ((members - last_centers[cluster_idx]) ** 2).sum()
        self.fitted = True
        return last_centers, inertia

    def fit(self, x):
        self.n_features_in_ = x.shape[1]
        self.inertia_ = float("inf")
        for i in range(self.n_init):
            centers = self._init_cluster_centers(x)  # (b, f)
            centers, inertia = self._kmeans_loop(x, centers)
            if inertia < self.inertia_:
                self.inertia_ = inertia
                self.cluster_centers_ = centers
        return self

    def predict(self, x):
        """Returns the cluster idx from given x"""
        if not self.fitted:
            raise RuntimeError("KMeans is not fitted")
        dist_sq = ((x[:, None, :] - self.cluster_centers_) ** 2).sum(-1)
        closest_idx = dist_sq.argmin(-1)  # (b,)
        return closest_idx

    def transform(self, x):
        """Returns distance from row in x to every cluster centers"""
        # (b, n_clusters)
        return np.sqrt(((x[:, None, :] - self.cluster_centers_) ** 2).sum(-1))
