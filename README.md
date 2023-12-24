# My Custom Machine Learning Algorithm Implementations

## Linear Regression

### Conditions for Linear Regression
- Linearity: the relationship between the independent and the dependent variables is linear
- Residual Independence: the error, or the residuals, of the model should be independent of one another
- No Multicollinearity: the independent variables themselves should not be highly correlated with another
- Homoscedasticity: the variance of the residuals should be relatively consistent throughout different magnitudes of independent variables

### In a Nutshell

Linear regression can be expressed in the form of $X\theta = y$. Our goal is to find such $\theta$ that minimizes the mean squared error, or MSE $=\frac{1}{N} \| \|y - X\theta\| \|_2^2$.


### Calculations and Algorithm

Algebraically, we see that MSE $\propto ||y-X\theta||_2^2 = (y-X\theta)^{\top}(y-X\theta) = y^{\top}y - 2y^{\top}X\theta + \theta^{\top}X^{\top}X\theta$.
Taking the derivative, we get $\frac{d}{d\theta}$ MSE $= -2X^{\top}y + 2X^{\top}X\theta$. In order for $\theta$ to minimize the MSE, we would need to have $\frac{d}{d\theta}$ MSE = 0.
In other words, we would need to have $-2X^{\top}y + 2X^{\top}X\theta = 0 \Rightarrow X^{\top}X\theta = X^{\top}y \Rightarrow \boxed{\theta = (X^{\top}X)^{-1}X^{\top}y}$.
Note that this is assuming the square matrix, $X^{\top}X$, is invertible, which is generally true if there are more observations than predictors and no perfect collinearity existing within the predictors.
Although this method of directly calculating $\theta$ for the linear regression may seem efficient, for $X$ with a large number of observations or predictors, the computing $(X^{\top}X)^{-1}$ is computationally expensive and therefore, may not be the most efficient solution to get the parameters of linear regression.

An alternative way, to find $\theta$ would be through gradient descent. The gradient $\frac{\partial}{\partial \theta}$ MSE $= \frac{1}{N} \frac{\partial}{\partial \theta} ||y - X\theta||_2^2 = -\frac{2}{N}X^{\top}y + \frac{2}{N}X^{\top}X\theta =$
$-\frac{2}{N}X^T(y-X\theta)$. Using this, we can first initialize the weights, $\theta$, and then update the weights using gradient descent $\Rightarrow \boxed{\theta \leftarrow \theta - \eta(-\frac{2}{N}X^{\top}(y-X\theta))}$, where $\eta$ is the learning rate of the gradient descent algorithm. Note that there must be a sufficient number of iterations to get the optimal $\theta$ as well as a small enough $\eta$ size to make sure the gradient descent converges to the optimal solution.

## Logistic Regression

### In a Nutshell
 Logistic regression is similar to linear regression and is used for binary classification, which can be extended to multiclass classification. Essentially, using an activation function, we transform the results of a linear regression problem into probabilities to classify data into binary categories. Typically, this activation function is the sigmoid function, $\sigma(z)=\frac{1}{1+e^{-z}}$. Logistic regression can be expressed as $P(Y=1|X)=\sigma(X\theta)$ or $P(Y=0|X)=1-\sigma(X\theta)$ where we attempt to minimize the binary cross-entropy loss function, which is $J(\theta)=-\frac{1}{m}\Sigma_{i=1}^m[y_i \log(\sigma(X_i\theta)) + (1 - y_i)\log(1-\sigma(X_i\theta))]$.


 ### Calculations and Algorithm

Algebracially, we will be doing gradient descent to achieve the least possible binary cross-entry loss, $J(\theta)$. The gradient $\frac{\partial}{\partial \theta} J(\theta) = -\frac{1}{m} \frac{\partial}{\partial \theta} \Sigma_{i=1}^m[y_i \log(\sigma(X_i\theta)) + (1 - y_i)\log(1-\sigma(X_i\theta))] = -\frac{1}{m}\Sigma_{i=1}^m[y_i\frac{1}{\sigma(X_i\theta)}\frac{\partial\sigma(X_i\theta)}{\partial \theta_j} + (1-y_i)\frac{1}{1-\sigma(X_i\theta)}(-\frac{\partial\sigma(X_i\theta)}{\partial \theta_j})] \Rightarrow$ we will note that $\sigma(z)=\frac{1}{1+e^{-z}}$ and that $\sigma'(z)=\sigma(z)(1-\sigma(z))$, which means that $\frac{\partial \sigma(X_i\theta)}{\partial \theta_j} = \sigma(X_i\theta)(1-\sigma(X_i\theta))X_{ij}$. Substituting this in, we get $\frac{\partial J(\theta)}{\partial \theta_j} = -\frac{1}{m}\Sigma_{i=1}^{m}[y_i(1-\sigma(X_i\theta))X_{ij}-(1-y_i)\sigma(X_i\theta)X_{ij}] = -\frac{1}{m}\Sigma_{i=1}^m[y_iX_{ij}-y_i\sigma(X_i\theta)X_{ij} - \sigma(X_i\theta)X_{ij} + y_i\sigma(X_i\theta)X_{ij}]$ $=\frac{1}{m} \Sigma_{i=1}^m[\sigma(X_i\theta) - y_i]X_{ij}$ $\Rightarrow \nabla_{\theta}J(\theta) = \frac{1}{m}X^{\top}(\sigma(X\theta) - y)$. Therefore, after first initializing the weights, we would update the parameters in the logistic regression using $\boxed{\theta \leftarrow \theta - \alpha (\frac{1}{m}X^{\top}(\sigma(X\theta)-y))}$, where $\alpha$ is the learning rate of the gradient descent algorithm. Note how similar the gradients for MSE and binary corss-entropy loss is with each other.
