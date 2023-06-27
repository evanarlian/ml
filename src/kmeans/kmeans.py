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
        # sklearn's k-means++ is always 1
        self.init = init
        self.n_init = 1 if self.init == "k-means++" else n_init
        self.max_iter = max_iter
        self.tol = tol
        self.fitted = False

    def _init_cluster_centers(self, x: np.ndarray):
        """Create the first clusters based on init"""
        if self.n_clusters > len(x):
            raise ValueError("`n_clusters` must be <= num samples")
        # x is (batch, feat) (b, f)
        if self.init == "random":
            chosen = np.random.choice(len(x), self.n_clusters, replace=False)
            return x[chosen]
        elif self.init == "k-means++":
            assert False, "not implemented"
        else:
            raise ValueError("init must be either `random` or `k-means++`")

    def _kmeans_loop(self, x, centers) -> tuple:
        """A single kmeans main loop, returns final centers and inertia"""
        # x is (b, f)
        # centers is (n_clusters, f)
        last_centers = centers.copy()
        for i in range(self.max_iter):
            # dist is (b, n_clusters)
            dist = np.sqrt(((x[:, None, :] - centers) ** 2).sum(-1))
            closest_idx = dist.argmin(-1)  # (b,)
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
        dist = np.sqrt(((x[:, None, :] - last_centers) ** 2).sum(-1))
        closest_idx = dist.argmin(-1)  # (b,)
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
        dist = np.sqrt(((x[:, None, :] - self.cluster_centers_) ** 2).sum(-1))
        closest_idx = dist.argmin(-1)  # (b,)
        return closest_idx

    def transform(self, x):
        """Returns distance from row in x to every cluster centers"""
        pass
