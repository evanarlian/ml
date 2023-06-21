import numpy as np


class LinearRegression:
    def __init__(self, lr: float = 0.01, epochs: int = 2000) -> None:
        self.lr = lr
        self.epochs = epochs
        self.fitted = False

    def fit(self, x, y):
        # prepare
        self.fitted = True
        x = x.reshape(-1, 1) if x.ndim == 1 else x  # (batch, m)
        y = y.reshape(-1, 1) if y.ndim == 1 else y  # (batch, n)
        self.w = np.random.standard_normal((x.shape[1], y.shape[1]))  # (m, n)
        self.b = np.random.standard_normal((y.shape[1],))  # (n,)
        n_samples = x.shape[0]

        for i in range(1, self.epochs + 1):
            # forward
            y_hat = x @ self.w + self.b  # (batch, n)
            error = y - y_hat  # (batch, n)
            loss = (error**2).mean()  # scalar
            if i % 100 == 0:
                print(f"{i} loss {loss}")
            # backward
            dw = x.T @ error / n_samples  # (m, n), x.T'ed to match w.shape
            db = error.sum(0) / n_samples  # (n,), error summed to match b.shape
            self.w += self.lr * dw
            self.b += self.lr * db
        return self

    def predict(self, x):
        if not self.fitted:
            raise RuntimeError("Estimator is not fitted")
        x = x.reshape(1, -1) if x.ndim == 1 else x  # (batch, n)
        y_hat = x @ self.w + self.b  # (batch, n)
        return y_hat
