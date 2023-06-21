# Linear regression

Types of linear regression
* Simple:
    * One independent variable and one dependent variable
    * $y = f(x)$
    * `nn.Linear(1, 1)`
* Multiple/multivariable:
    * Many independent variables and one dependent variable
    * $y = f(x_1, x_2, ..., x_n)$
    * `nn.Linear(n, 1)`
* Multivariate:
    * Many independent variables and many dependent variables
    * $y_1, y_2, ..., y_m = f(x_1, x_2, ..., x_n)$
    * `nn.Linear(n, m)`


# References
* [Linear regression types](https://stats.stackexchange.com/questions/2358/explain-the-difference-between-multiple-regression-and-multivariate-regression)
>