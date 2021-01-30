<!-- TOC -->

    - [Lecture 1: Introduction Probability Theory](#lecture-1-introduction-probability-theory)
        - [Terminologies](#terminologies)
        - [Supervised v.s. Unsupervised](#supervised-vs-unsupervised)
        - [Evaluation](#evaluation)
    - [Lecture 2: Statistical Schools of Thoughts](#lecture-2-statistical-schools-of-thoughts)
        - [Frequentist statistics](#frequentist-statistics)
        - [Decision Theory](#decision-theory)
        - [Risk & Empirical Risk Minimization (ERM)](#risk--empirical-risk-minimization-erm)
        - [Mean Squared Error (of parameter estimator)](#mean-squared-error-of-parameter-estimator)
        - [Bayesian Statistics](#bayesian-statistics)
        - [Categories of Probabilistic Models](#categories-of-probabilistic-models)
    - [Lecture 3: Linear Regression & Optimisation](#lecture-3-linear-regression--optimisation)
        - [Linear Regression](#linear-regression)
        - [Optimization](#optimization)
        - [Coordinate descent](#coordinate-descent)
        - [Gradient descent](#gradient-descent)
        - [Convex objective functions](#convex-objective-functions)
        - [$L_1$ and $L_2$ norms](#l_1-and-l_2-norms)
        - [Linear Regression optimization: Least Square Method](#linear-regression-optimization-least-square-method)
    - [Lecture 4: Logistic Regression & Basis Expansion](#lecture-4-logistic-regression--basis-expansion)
        - [Logistic Regression](#logistic-regression)
        - [Linear v.s. logistic probabilistic models](#linear-vs-logistic-probabilistic-models)
        - [Logistic MLE](#logistic-mle)
        - [Information divergence (extra)](#information-divergence-extra)
        - [Cross entropy](#cross-entropy)
        - [Training as cross-entropy minimisation](#training-as-cross-entropy-minimisation)
        - [Basis Expansion (Data transformation)](#basis-expansion-data-transformation)
- [Radial basis functions (RBFs)](#radial-basis-functions-rbfs)
    - [Challenges of basis expansion (data transformation)](#challenges-of-basis-expansion-data-transformation)
    - [Taking the idea of basis expansion to the next level](#taking-the-idea-of-basis-expansion-to-the-next-level)
  - [Lecture 5: Regularisation](#lecture-5-regularisation)
    - [Irrelevant / Multicolinear features](#irrelevant--multicolinear-features)
    - [Ill-posed problems](#ill-posed-problems)
    - [Re-conditioning the problem (Ridge Regression)](#re-conditioning-the-problem-ridge-regression)
    - [Regulariser as a prior (Bayesian intepretation of ridge regression)](#regulariser-as-a-prior-bayesian-intepretation-of-ridge-regression)
    - [Regularisation in non-linear models](#regularisation-in-non-linear-models)
    - [Explicit model selection](#explicit-model-selection)
    - [Regularisation](#regularisation)
    - [Regulariser as a constraint](#regulariser-as-a-constraint)
    - [Closed form solutions](#closed-form-solutions)
    - [Bias-variance trade-off](#bias-variance-trade-off)
  - [Lecture 6: Perceptron](#lecture-6-perceptron)
    - [Artificial Neural Network (RNN, CNN, MLP)](#artificial-neural-network-rnn-cnn-mlp)
    - [Perceptron](#perceptron)
    - [Stochastic gradient descent (SGD)](#stochastic-gradient-descent-sgd)
    - [Perceptron training algorithm (SDG with batch size 1)](#perceptron-training-algorithm-sdg-with-batch-size-1)
  - [Lecture 7: Multilayer Perceptron (MLP) Backpropagation](#lecture-7-multilayer-perceptron-mlp-backpropagation)
    - [Multilayer Perceptron](#multilayer-perceptron)
    - [ANN](#ann)
    - [Loss function for ANN training](#loss-function-for-ann-training)
    - [SGD for ANN](#sgd-for-ann)
    - [Backpropagation (for updating weights)](#backpropagation-for-updating-weights)
    - [Forward propagation (just compute outputs for each layer, make predictions)](#forward-propagation-just-compute-outputs-for-each-layer-make-predictions)
    - [Further notes on ANN](#further-notes-on-ann)
  - [Lecture 8: Deep learning, CNN, Autoencoders](#lecture-8-deep-learning-cnn-autoencoders)
    - [Deep learning](#deep-learning)
    - [Representation learning](#representation-learning)
    - [Depth v.s. width](#depth-vs-width)
    - [Convolutional Neural Networks](#convolutional-neural-networks)
    - [Components of CNN](#components-of-cnn)
    - [Downsampling via max pooling](#downsampling-via-max-pooling)
    - [Convolution as a regulariser](#convolution-as-a-regulariser)
    - [Autoencoder](#autoencoder)
    - [Bottleneck](#bottleneck)
    - [Dimensionality reduction](#dimensionality-reduction)
  - [Lecture 9: Support Vector Machine](#lecture-9-support-vector-machine)
    - [Linear hard-margin SVM](#linear-hard-margin-svm)
    - [SVM vs. Perceptron](#svm-vs-perceptron)
    - [Separation boundary](#separation-boundary)
    - [Margin width](#margin-width)
    - [SVM Parameters](#svm-parameters)
    - [SVM: Finding separating boundary](#svm-finding-separating-boundary)
    - [SVM as regularised ERM](#svm-as-regularised-erm)
  - [Lecture 10: Soft-Margin SVM, Lagrangian Duality](#lecture-10-soft-margin-svm-lagrangian-duality)
    - [Soft-Margin SVM](#soft-margin-svm)
    - [Hinge loss: soft-margin SVM loss](#hinge-loss-soft-margin-svm-loss)
    - [Soft-Margin SVM Objective](#soft-margin-svm-objective)
    - [Two variations of SVM](#two-variations-of-svm)
    - [Constraint optimisation](#constraint-optimisation)
    - [Lagrangian and duality](#lagrangian-and-duality)
    - [Dual program for hard-margin SVM](#dual-program-for-hard-margin-svm)
    - [Making predictions with dual solution](#making-predictions-with-dual-solution)
    - [Optimisation for Soft-margin SVM](#optimisation-for-soft-margin-svm)
    - [Complementary slackness](#complementary-slackness)
    - [Training SVM](#training-svm)
  - [Lecture 11: Kernel Methods](#lecture-11-kernel-methods)
    - [Kernelising the SVM](#kernelising-the-svm)
    - [Kernel representation](#kernel-representation)
    - [Approaches to non-linearity](#approaches-to-non-linearity)
    - [Modular learning](#modular-learning)
    - [Constructing kernels](#constructing-kernels)
  - [Lecture 12: Ensemble methods](#lecture-12-ensemble-methods)
    - [Combining models (Ensembling)](#combining-models-ensembling)
    - [Bagging (bootstrap aggregating)](#bagging-bootstrap-aggregating)
    - [Using out-of-sample data](#using-out-of-sample-data)
    - [Boosting](#boosting)
    - [Adaboost](#adaboost)
    - [Bagging v.s. Boosting](#bagging-vs-boosting)
    - [Stacking](#stacking)
  - [Lecture 13: Multi-armed bandits](#lecture-13-multi-armed-bandits)
    - [Stochastic multi-armed bandits](#stochastic-multi-armed-bandits)
    - [Exploration v.s. Exploitation](#exploration-vs-exploitation)
    - [Stochastic MAB setting](#stochastic-mab-setting)
    - [Greedy](#greedy)
    - [$\epsilon$-Greedy](#epsilon-greedy)
    - [Upper Confidence Bound (UCB)](#upper-confidence-bound-ucb)
  - [Lecture 14: Bayesian regression](#lecture-14-bayesian-regression)
    - [Bayesian Inference](#bayesian-inference)
    - [Frequentist v.s. Bayesian](#frequentist-vs-bayesian)
    - [Bayesian Regression](#bayesian-regression)
    - [Conjugate Prior](#conjugate-prior)
    - [Stages of Training](#stages-of-training)
    - [Prediction with uncertain $w$](#prediction-with-uncertain-w)
    - [Caveats (Notes)](#caveats-notes)
  - [Lecture 15: Bayesian classification](#lecture-15-bayesian-classification)
    - [Discrete Conjugate prior](#discrete-conjugate-prior)
    - [Suite of useful conjugate priors](#suite-of-useful-conjugate-priors)
    - [Bayesian Logistic Regression](#bayesian-logistic-regression)
  - [Lecture 16: PGM Representation](#lecture-16-pgm-representation)
    - [PGM](#pgm)
    - [Bayesian statistical learning v.s. PGM (aka. "Bayes Nets")](#bayesian-statistical-learning-vs-pgm-aka-bayes-nets)
    - [Joint distribution](#joint-distribution)
    - [Independence](#independence)
    - [Factoring Joint Distributions](#factoring-joint-distributions)
    - [Directed PGM](#directed-pgm)
    - [Plate notation](#plate-notation)
    - [PGM: frequentist v.s. Bayesian](#pgm-frequentist-vs-bayesian)
    - [Undirected PGMs](#undirected-pgms)
    - [Undirected PGM formulation](#undirected-pgm-formulation)
    - [Directed to undirected](#directed-to-undirected)
    - [Why U-PGM](#why-u-pgm)
    - [PGM examples](#pgm-examples)
  - [Lecture 17: PGM Probabilistic and Statistical Inference](#lecture-17-pgm-probabilistic-and-statistical-inference)
    - [Probabilistic inference on PGMs](#probabilistic-inference-on-pgms)
    - [Elimination algorithm](#elimination-algorithm)
    - [Statistical inference on PGMs](#statistical-inference-on-pgms)
  - [Lecture 18: Gaussian Mixture Model, Expectation Maximization](#lecture-18-gaussian-mixture-model-expectation-maximization)
    - [Unsupervised learning](#unsupervised-learning)
    - [Gaussian Mixture Model (GMM)](#gaussian-mixture-model-gmm)
    - [Clustering as model estimation](#clustering-as-model-estimation)
    - [Fitting the GMM](#fitting-the-gmm)
    - [Expectation Maximisation (EM) algorithm](#expectation-maximisation-em-algorithm)
    - [Algorithm (Simple version)](#algorithm-simple-version)
    - [EM for GMM and generally](#em-for-gmm-and-generally)
    - [Not-Examinable Part](#not-examinable-part)
    - [Estimating Parameters of GMM](#estimating-parameters-of-gmm)
    - [K-means as a EM for a restricted GMM](#k-means-as-a-em-for-a-restricted-gmm)
  - [Lecture 19: Dimensionality Reduction](#lecture-19-dimensionality-reduction)
    - [Dimensionality reduction](#dimensionality-reduction-1)
    - [Pricinpal component analysis (PCA)](#pricinpal-component-analysis-pca)
    - [Formulating the problem](#formulating-the-problem)
    - [PCA](#pca)
    - [Efficient algorithm for PCA](#efficient-algorithm-for-pca)
    - [Linear regression v.s. PCA](#linear-regression-vs-pca)
    - [Additional effect of PCA](#additional-effect-of-pca)
    - [Non-linear data and kernel PCA](#non-linear-data-and-kernel-pca)

<!-- /TOC -->
### Lecture 1: Introduction Probability Theory

#### Terminologies
* **Instance**: measurements about individual entities/objects (no label)
* **Attributes**: component of the instances
* **Label**: an outcome that is categorical, numerical, etc.
* **Examples**: instance coupled with label
* **Models**: discovered relationship between attributes and/or label

#### Supervised v.s. Unsupervised
* **Supervised**: 
  * Labelled data
  * Predict labels on new instances
* **Unsupervised**: 
  * Unlabelled data
  * Cluster related instances; project to fewer dimensions; understand attribute relationships

#### Evaluation
1. Pick an evaluation metric comparing label v.s. prediction
   * E.g. Accuracy, Contingency table, Precision-Recall, ROC curves
2. Procure an independent, labelled test set
3. "Average" the evaluation metric over the test set
(When data poor, use cross-validation)

Probability相关的部分就不写了

---

### Lecture 2: Statistical Schools of Thoughts

#### Frequentist statistics
* Unknown params are treated as having fixed but unknown values
* Parameter estimation:
  * Classes of models $\{ p_{\theta} (x): \theta \in \Theta\}$ indexed by parameters $\Theta$
  * Point estimate $\hat{\theta}(x_1, ..., x_n)$: a function (or statistic) of data (samples)
    * A single value as an estimate of a population parameter
* If $\hat{\theta}$ is an estimator for $\theta$
  * Bias: $Bias_{\theta}(\hat{\theta}) = E_{\theta}[\hat{\theta}] - \theta$
  * Variance: $Var_{\theta}(\hat{\theta}) = E_{\theta}[(\hat{\theta} - E_{\theta}[\hat{\theta}])^2]$
* Asymptotic properties:
  * Consistency: $\hat{\theta} \rightarrow \theta$ (converges in probability) as $n \rightarrow \infty$
  * Efficiency: asymptotic variance is as small as possible (reach Cramer-Rao lower bound)
* Maximum-Likelihood Estimation (MLE)
  * General principle for designing estimators
  * Involves optimisation
  * $\hat{\theta} \in \argmax_{\theta \in \Theta} \prod_{i=1}^{n} p_{\theta}(x_i)$
  * MLE estimators are consistent (but usually biased)
  * "Algorithm":
    1. Given data $X_1, ..., X_n$
    2. Likelihood: $L(\theta) = \prod_{i=1}^{n} p_{\theta}(X_i)$
    3. Optimise to find best params
        * Take partial derivatives of log likelihood: $l'(\theta)$
        * Solve $l'(\theta) = 0$
        * If fail, use iterative gradient method (e.g. fisher scoring)

#### Decision Theory
* Decision rule: $\delta(x) \in A$ (action space)
  * E.g. point estimate, out-of-sample prediction
* Loss function $l(a, \theta)$: economic cost, error metric
  * E.g. square loss $(\hat{\theta} - \theta)^2$, 0-1 loss $I(y \neq \hat{y})$

#### Risk & Empirical Risk Minimization (ERM)
* In decision theory, really care about **expected loss**
* **Risk** : $R_\theta[\delta] = E_{X \sim \theta}[l(\delta(X), \theta)]$
  * Risk = Expected Loss
  * aka. Generalization error
* **Goal**: Choose $\delta$ (decision) to minimise $R_\theta[\delta]$
  * Can't calculate risk directly
  * Don't know the real distribution the samples comes from, therefore don't now $E(X)$
* **ERM**
  * Use training set X to approximate $p_\theta$ (Empirical)
  * Minimise empirical risk $\hat{R}_\theta[\delta] = \frac{1}{n} \sum_{i=1}^n l(\delta(X_i), \theta)$

#### Mean Squared Error (of parameter estimator)
* Bias-variance decomposition of **square-loss risk**
* $E_{\theta}[(\theta - \hat{\theta})^2] = [Bias(\hat{\theta})]^2 + Var_{\theta}(\hat{\theta})$

#### Bayesian Statistics
* Unknown params have associated distributions reflecting prior **belief**
* Prior distribution $P(\theta)$
  * Params are modeled like r.v.'s
  * Data likelihood $P_{\theta}(X)$ written as conditional $P(X|\theta)$
* Rather than point estimate $\hat{\theta}$
  * Bayesians update prior belief $P(\theta)$ with observed data to the posterior distribution: $P(\theta | X)$
* Bayesian probabilistic inference
  1. Start with prior $P(\theta)$ and likelihood $P(X|\theta)$
  2. Observe data $X = x$
  3. Update prior to posterior $P(\theta | X = x)$
* Primary tools to obtain the posterior
  * Bayes Rule: reverse order of conditioning
    * $P(\theta | X = x) = \frac{P(X = x | \theta)P(\theta)}{P(X = x)}$
  * Marginalization: eliminates unwanted variables
    * $P(X = x) = \sum_t P(X = x, \theta = t)$
* Bayesian point estimation common approaches
  * Posterior mean
    * $E_{\theta | X}[\theta] = \int \theta P(\theta | X) d\theta$
  * Posterior mode (MAP)
    * $\argmax_\theta P(\theta | X)$
* MLE in Bayesian context
  * MLE = MAP if using uniform prior $P(\theta)$
  * (No prior belief about $\theta$)

#### Categories of Probabilistic Models
* Parametric v.s. Non-Parametric
  1. Parametric
     * Determined by fixed, finite number of parameters
     * Limited flexibility
     * Efficient statistically and computationally
  2. Non-Parametric
     * Number of parameters grows with data, potentially infinite
     * More flexible
     * Less efficient
* Generative v.s. Discriminative
  1. Generative
     * Model full joint $P(X,Y)$
     * E.g. Naive Bayes
  2. Discriminative
     * Model conditional $P(Y|X)$ only (directly)
     * E.g. Linear Regression

### Lecture 3: Linear Regression & Optimisation

#### Linear Regression
* Assume a probabilistic model
    * $y = X\beta + \epsilon$
* Assume Gaussian noise (independent of X):
    * $\epsilon \sim N(0, \sigma^2)$
* Discriminative model
    * $p(y|\mathbf{x}) = \frac{1}{\sqrt{2\pi\sigma^2}}\text{exp}(-\frac{(y-\mathbf{x}\beta)^2}{2\sigma^2})$
* Unknown param: $\beta$ (and $\sigma^2$)
* MLE: choose param values that maximise the probability of observed data (likelihood)
  * "Log trick": instead of maximising likelihood, we can maximise log-likelihood (since log is strictly monotonic increasing)
* Under this model ("normal" linear regression):
  * MLE is equivalent to minimising SSE (or RSS)

#### Optimization
* Training = Fitting = Parameter estimation
* Typical formulation (minimise loss = objective)
  * $\hat{\theta} \in \argmin_{\theta \in \Theta} L(data, \theta)$
* Analytic (aka. closed form) solution
  * 1st order derivatives for optimality:
    * $\frac{\partial L}{\partial \theta_1} = ... = \frac{\partial L}{\partial \theta_p} = 0$
  * (Need to check 2nd derivative)
* Approximate iterative solution (e.g. IRWLS)
  * Initialisation: choose starting guess $\mathbf{\theta}^{(1)}$, set $i=1$
  * Update: $\theta^{(i+1)} \leftarrow SomeRule[\theta^{(i)}]$, set $i \leftarrow i+1$
  * Termination: decide whether to stop
  * Go to step 2
  * Stop: return $\hat{\theta} \approx \theta^{(i)}$

#### Coordinate descent
* 一次 update 一个 $\theta_i$
* Suppose $\theta = [\theta_1, ..., \theta_K]^{T}$
  1. Choose $\theta^{(1)}$ and some $T$
  2. For $i$ from $1$ to $T$ (update all params $T$ times)
     * $\theta^{(i+1)} \leftarrow \theta^{(i)}$ (copy param values)
     * For $j$ from 1 to $K$: (update one param each time)
        * Fix components of $\theta^{(i+1)}$, except j-th component
        * Find $\hat{\theta}_j^{(i+1)}$ that minimises $L(\theta_j^{(i+1)})$
        * Update j-th component of $\theta^{(i+1)}$
  3. Return $\hat{\theta} \approx \theta^{(i)}$
* (Other stopping criteria can be used)

#### Gradient descent
* Gradient denoted as $\nabla L = [\frac{\partial L}{\partial \theta_1}, ..., \frac{\partial L}{\partial \theta_p}]^{T}$
* Algorithm: 
  1. Choose $\theta^{(1)}$ and some $T$
  2. For $i$ from $1$ to $T^*$
     * update: $\theta^{(i+1)} = \theta^{(i)} - \eta \nabla L (\theta^{(i)})$
  3. Return $\hat{\theta} \approx \theta^{(i)}$
* Note: $\eta$ (learning rate) is dynamically updated in each step
* Variants: SGD, mini batches, momentum, AdaGrad

#### Convex objective functions
* "Bowl" shaped functions
* Every local min is global min
* Informal definition: line segment between any two points on graph of function lies above or on the graph
* Gradient descent on (strictly) convex function guaranteed to find a (unique) global minimum

#### $L_1$ and $L_2$ norms
* Norm: length of vectors
* $L_2$ norm (aka. Euclidean distance)
  * $||a|| = ||a||_2 \equiv \sqrt{a_1^2 + ... + a_n^2}$
* $L_1$ norm (aka. Manhattan distance)
  * $||a||_1 \equiv |a_1| + ... + |a_n|$
* E.g. Sum of squared errors:
  * $L = \sum_{i=1}^n (y_i - \sum_{j=0}^{m}X_{ij}\beta_{j})^2 = ||y - X\beta||^2$

#### Linear Regression optimization: Least Square Method
* To find $\beta$, minimize the **sum of squared errors**:
  * $SSE/RSS = \sum_{i=1}^n (y_i - \sum_{j=0}^m X_{ij}\beta_{j})^2$
* Setting derivative to zero and solve for $\beta$: (normal equation)
  * $b = (X^TX)^{-1}X^{T}y$
  * Well defined only if the inverse exists

---

### Lecture 4: Logistic Regression & Basis Expansion

#### Logistic Regression
* Why not linear regression for classification?
  * Predict "Yes" if $s \geq 0.5$
  * Predict "No" if $s < 0.5$
  * ($s = x\hat{\beta}$, estimated probability for class 1 given a data point)
  * Reason:
    * Can be susceptible (易受影响) to outliers
    * least-squares criterion looks **unnatural** in this setting
* Problem: the probability needs to be between 0 and 1
* Logistic function
  * $f(s) = \frac{1}{1+\text{exp}(-s)}$
* Logistic regression model:
  * $P(Y=1|x) = \frac{1}{1+\text{exp}(-x^T\beta)}$
  * In GLM:
    * Mean: $\mu = P(Y=1 | x)$
    * Link function: $g(\mu) = log \frac{P(Y=1 | x)}{P(Y=0 | x)} = \eta = x^T\beta$ (log odds)
  * (Can use Cross-Validation to choose the threshold, usually just use 0.5)
* Logistic regression is a linear classifier
  * Logistic regression model:
    * $P(Y = 1 | x) = \frac{1}{1 + \text{exp}(-x^T\beta)}$
  * Classification rule:
    * Class "1" 
      * If $P(Y=1 | x) = \frac{1}{\exp(-x^{T}\beta)} > \frac{1}{2}$
      * Else class "0"
    * Decision boundary (line): 
      * $P(Y = 1 | x) = \frac{1}{1 + \text{exp}(-x^T\beta)} = \frac{1}{2}$
      * Equivalently, $P(Y = 0 | x) = P(Y = 1 | x)$
    * (In higher dimensional problems, the decision boundary is a plane or hyperplane, vector $\beta$ is perpendicular to the decision boundary)

#### Linear v.s. logistic probabilistic models
* Linear regression
  * Assume $\epsilon \sim N(0, \sigma^2)$
  * Therefore assume $y \sim N(X\beta, \sigma^2)$
* Logistic regression
  * Assume $y \sim Bernoulli(p = logistic(x^{T}\beta))$

#### Logistic MLE
* Doesn't have closed form solution (cannot solve $\frac{\partial L}{\partial \beta} = 0$ directly)
* Therefore use iterative optimisation
  * E.g. Newton-Raphson, IRWLS, or gradient descent
* Good news: it's a convex problem (if no irrelevant features) $\Rightarrow$ guaranteed to get global minimum

#### Information divergence (extra)
* To **compare models** with different num of params in an all-subsets search
* May use information theoretic criterion which estimates information divergence between true model and a given candidate model (working model)
* Best model: **smallest** criterion value
* E.g. Kullback-Leibler Information 
  * $KL(f_1,f_2) = E_{f_1}[log \frac{f_2}{f_1}] = \int_{x} log \frac{f_2}{f_1}f(x) dx$
  * Problem: don't know the true model $\Rightarrow$ cannot compute $KL$
  * E.g. two binomial distribution: one for working model, one for true model
    * Choose the model minimise $KL$

#### Cross entropy
* A method for comparing two distiributions
* A measure of divergence between reference distribution $g_{ref}(a)$ and estiamted distribution $g_{est}(a)$
  * For discrete distribution:
    * $H(g_{ref}, g_{est}) = -\sum_{a \in A} g_{ref}(a) log \, g_{est}(a)$

#### Training as cross-entropy minimisation
* Consider log-likelihood for a single data point
    * $log \, p(y_i | x_i) = y_i log(\theta(x_i)) + (1 - y_i) log(1 - \theta(x_i))$
* Negative cross entropy
  * $H(g_{ref}, g_{est}) = -\sum_a g_{ref}(a) log \, g_{est}(a)$
* Reference (true) distribution:
  * $g_{ref}(1) = y_i$ and $g_{ref}(0) = 1 - y_i$
* Estimate true distribution as:
  * $g_{est}(1) = \theta(x_i)$ and $g_{est}(0) = 1 - \theta(x_i)$
* Find $\beta$ that minimise sum of cross entropies per training point

#### Basis Expansion (Data transformation)
* Extend the utility of models via **data transformation**
* For linear regression:
  * Transformation data $\Rightarrow$ make data more linear!
  * Map data onto another feature space s.t. data is linear in that space  
    * $\mathbf{x}$: the original set of features
    * $\phi$: $\R^m \rightarrow \R^k$ denotes transformation
    * $\phi(\mathbf{x})$: new feature set
* Polynomial regression:
  * New features:
    * $\phi_1(x) = x$
    * $\phi_2(x) = x^2$
  * Quadratic regression (linear in new feature set): 
    * $y = w_0 + w_1 \phi_1(x) + w_2 \phi_2(x) = w_0 + w_1 x + w_2 x^2$
* Can be applied for both regression and classification
* There are many possible choices of $\phi$
* Binary classification:
  * If dataset not linearly separable (non-linear problem)
  * Define transformation as:
    * $\phi_i(x) = ||x - z_i||$ (euclidean distance)
      * where $z_i$ is some pre-defined **constants**
      * Distances to each $z_i$ as new features

## Radial basis functions (RBFs)
* Motivated from approximiation theory
* Sums of RBFs are used to **approximate given functions**
* Radial basis functions:
  * $\phi(x) = \psi(||x - z||)$, where $z$ is a constant
* Examples:
  * $\phi(x) = ||x - z||$
  * $\phi(x) = \text{exp}(-\frac{1}{\sigma} ||x-z||^2)$

#### Challenges of basis expansion (data transformation)
* One limitation: the transformation needs to be defined beforehand
  * Need to choose the **size** of new feature set
  * If using  RBFs, need to choose $z_i$
* Choosing $z_i$:
  1. Uniformly spaced points (grids)
  2. Cluster training data and use cluster centroids
  3. Use training data $z_i \equiv x_i$ (some $x_i$)
    * E.g. $\phi_i(x) = \psi(||x - x_i||)$
    * For large datasets, this results in a **large number of features**

#### Taking the idea of basis expansion to the next level
* One idea: to learn the transformation $\phi$ from data
  * E.g. Artificial Neural Networks
* Another extension: use kernel trick
* In **sparse kernel machines**, training dependes ony on a few data points
  * E.g. SVM


### Lecture 5: Regularisation

#### Irrelevant / Multicolinear features
* Co-linearity between features
* Features not linearly independent
  * E.g. If $x_1$ and $x_2$ are the same $\Rightarrow$ perfectly correlated
* For linear model, feature $X_j$ is irrelevant if
  * $X_j$ is a linear combination of other columns
  $$
    X_{.j} = \Sigma_{l \neq j} \alpha_{l}X_{.l}
  $$
  for some scalars $\alpha_l$
  * Equivalently: Some eigenvalue of $X'X$ is zero
* Problems
  1. The solution is not **unique**
     * Infinite number of solutions
  2. Lack of interpretability
     * cannot interpret the weights
  3. Optimising to learn parameter is **ill-posed problem**


#### Ill-posed problems
* Well-posed problem
  1. a solution exists
  2. the solution is unique
  3. the solution's behavior changes continuously with the initial condition
* If ill-posed, there is **no closed form solution**
  * Closed form solution $\hat{w} = (X'X)^{-1}X'y$
  * But if irrelevant, $X'X$ has no inverse (singular)
  * (Even near-irrelevance / colinearity can be problematic)

#### Re-conditioning the problem (Ridge Regression)
* Make it a well-posed solution and also prevent fitting noise / overfitting
* Original problem: minimise squared error
$$
    || y - Xw ||_2^2
$$ 
* Regularised problem (L2, **Ridge regression**): minimise
$$
    || y - Xw ||_2^2 + \lambda||w||_2^2 \text{ for } \lambda > 0
$$
  * Turns the ridge into a peak ($\Rightarrow$ unique solution)
  * Adds $\lambda$ to eigenvalues of $X'X$: makes invertible

#### Regulariser as a prior (Bayesian intepretation of ridge regression)
* Let prior distribution be:
$$
  W \sim N(0, 1/\lambda)
$$
* Higher $\lambda$, more confidence at prior, therefore ignore data more
* Computing posterior and take **MAP**
$$
  \text{log}(posterior) = \text{log}(likelihood) + \text{log}(prior) - \text{log}(marg)
$$
  * can just ignore $\text{log}(marg)$, since this term doesn't affect optimisation
* Arrive at the problem of minimising:
$$
  || y - Xw ||_2^2 + \lambda||w||_2^2
$$
* Become equivalent problem: Ridge Regression

---

#### Regularisation in non-linear models
* There is trade-off between **overfitting** and **underfitting**
* Right model class $\Theta$ will sacrifice some traininig error, for test error
* Choosing model complexity (2 Methods)
  1. Explicit model selection
       * Choosing degree of polynomial model by CV or held out validation
  2. Regularisation

#### Explicit model selection
* Using hold-out or CV to select the model
1. Split data into $D_{train}$ and $D_{validate}$ sets
2. For each degree d (# of parameters), we have model $f_d$
   1. Train $f_d$ on $D_{train}$
   2. Test (evaluate) $f_d$ on $D_{validate}$
3. Pick degree $\hat{d}$ that gives the best test score
4. Re-train model $f_{\hat{d}}$ using all data (return this final model)


#### Regularisation
* Solving the problem:
$$
  \hat{\theta} \in \argmin_{\theta \in \Theta} (L(data, \theta) + \lambda R(\theta))
$$
  * E.g. Ridge regression
  $$ 
    \hat{w} \in \argmin_{w \in W} ||y - Xw||_2^2 + \lambda||w||_2^2
  $$
* Note: regulariser $R(\theta)$ doesn't depend on data
* Use held-out validation / cross validation to choose $\lambda$

#### Regulariser as a constraint
* Modified problem: 
  * minimise $||y - Xw||_2^2$ subject to $||w||_2^2 \leq \lambda$ for $\lambda > 0$ 
  <img src="pic/constraint.png" width="500">
  * $w^*$ is the solution
  * Lasso encourages solution to sit on the axes
    * Some of the weights are set to zero $\Rightarrow$ solution is sparse

#### Closed form solutions
1. Linear regression
  $$
    (X'X)^{-1}X'y
  $$
2. Ridge regression
  $$
    (X'X + \lambda I )^{-1}X'y
  $$
3. Lasso
   * No closed-form solution, but solutions are sparse and suitable for high-dim data

#### Bias-variance trade-off
* Model complexity is a major factor that influences the ability of the model to **generalise**
* Bias-variance decomposition
  * Risk / test error = $Bias^2 + Variance + Irreducible Error (noise)$
$$
  E[l(Y, \hat{f}(X_0))] = E[(Y - \hat{f})^2] = (E[Y] - E[\hat{f}])^2 + Var[\hat{f}] + Var[Y]
$$
* Squared loss for supervised-regression
$$
  l(Y, \hat{{f}(X_0)}) = (Y - \hat{f}(X_0))^2
$$
* Simple model $\Rightarrow$ high bias, low variance
* Complex model $\Rightarrow$ low bias, high variance

---

### Lecture 6: Perceptron

#### Artificial Neural Network (RNN, CNN, MLP)
* A network of processing elements
  * Each element converts inputs to output
  * Output: a (activation) function of a **weighted sum** of inputs (linear combination)
* To use ANN, we need
  * To design network topology
  * Adjust weights to given data
* Training an ANN $\Rightarrow$ adjusting weights for training data given a pre-defined network topology

#### Perceptron
* Perceptron is a binary classifier
  * $s = \sum_{i = 0}^m x_i w_i$
    * Predict class A if $s \geq 0$
    * Predict class B if $s < 0$
* Loss function
  * Usually don't use 0-1 loss for training, since cannot calculate the gradient
  * Shouldn't give penalty for correctly classified examples
    * Penalty (loss) = $s$
  * Can be re-written as $L(s,y) = max(0,-sy)$
<img src="pic/perceptron.png" width="400">

#### Stochastic gradient descent (SGD)
* Stochastic = Random: shuffling training examples
* Randomly shuffle / split all training examples in B batches
* Choose initial $\theta^{(1)}$
* For $i$ from 1 to $T$ (epochs)
   * For $j$ from 1 to $B$ (batches)
      * Do gradient descent update **using data from batch $j$**
* Advantage: computational feasibility for large datasets

#### Perceptron training algorithm (SDG with batch size 1)
* Choose initial guess $w^{(0)}$, $k=0$
* For $i$ from 1 to $T$ (epochs)
  * For $j$ from 1 to $N$ (training examples)
    * Consider examples $\{\mathbf{x}_j, y_j \}$
    * Update: $w^{(k++)} = w^{(k)} - \eta \nabla L(w^{(k)})$
* Training rule (gradient descent)
  * Correct: We have $\frac{\partial L}{\partial w_i} = 0$ when $sy > 0$
  * Misclassified: We have $\frac{\partial L}{\partial w_i} = -x_i$ when y = 1 and $s < 0$
  * Misclassified: We have $\frac{\partial L}{\partial w_i} = x_i$ when y = -1 and $s > 0$
  * For sy = 0, we can do either of these (doesn't matter)
* When classified correctly, weights are unchanged
* When misclassified, update: $\mathbf{w}^{(k+1)} += -\eta(\pm \mathbf{x})$
  * $\pm x$ is gradient
* **Convergence theorem:** if the trianing data is linearly separable, the algorithm is guaranteed to **converge to a solution**. This is, there exist a finite $K$ s.t. $L(\mathbf{w}^K) = 0$
* **Pros and cons:**
  * Pros: if data is linearly separable, the perceptron trianing algorithm will converge to a correct solution
  * Cons: 
    * There are infinitely many possible solutions
    * If the data is not linearly separable, the training will fail completely rather than give some approx. solution


### Lecture 7: Multilayer Perceptron (MLP) Backpropagation

#### Multilayer Perceptron
* Modelling non-linearity via **function composition (构成)**
* Linear models cannot solve non-linearly separable problems
  * Possible solution: composition (combine neurons)
* Perceptron: a building block for ANN
* Not restricted to binary classification, there're various **activation functions**:
  * Step function
  * Sign function
  * Logistic function
  * Tanh function
  * Rectifier
  * ...

#### ANN
* Can be naturally adapted to various supervised learning setups (e.g. univariate/multivariate regression, binary/multivariate classification)
* Capable of approximating plethora non-linear functions
* **Universal approximation theorem:** (就是说hidden layer可以approximate各种continuous function)
  * **an ANN with a hidden layer** with a finite number of units, and mild assumptions on the activation function, can approximate continuous functions on compact subsets of $R^n$ arbitrarily well
* To train your network:
  * Define the loss function and find params that minimise the loss on training data (e.g. use SDG)
* Loss function for ANN:
  * As regression, can use **squared error**

#### Loss function for ANN training
$$ L = \frac{1}{2} (\hat{f}(x, \theta) - y)^2 = \frac{1}{2} (z-y)^2 $$
  * Training: minimise $L$ w.r.t. $\theta$
    * Fortunately $L(\theta)$ is differentiable
    * Unfortunately no analytic solution in general

#### SGD for ANN
* Choose initial guess $\theta^{(0)}, k = 0$
   * Here $\theta$ is all set of weights from all layers
* For $i$ from 1 to $T$ (epochs)
   * For $j$ from 1 to $N$ (training examples)
      * Consider examples $\{\mathbf{x}_j, y_j \}$
      * Update: $\theta^{(i+1)} = \theta^{(i)} - \eta \nabla L(\theta^{(i)})$

---

#### Backpropagation (for updating weights)
* Calculate **gradient of loss** of a composition
* Recall that the output $z$ of an ANN is a function composition, and hence $L(z)$ (loss) is also a composition
$$ L = 0.5(z-y)^2 = 0.5(h(s) - y)^2 = 0.5(s-y)^2 $$
* Backpropagation makes use of this fact by applying the **chain rule** for derivatives (从后往前一层一层differentiate回去)

$$ \frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial s} \frac{\partial s}{\partial w_j}$$

$$ \frac{\partial L}{\partial v_{ij}} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial s} \frac{\partial s}{\partial u_j} \frac{\partial u_j}{\partial r_j} \frac{\partial r_j}{\partial v_{ij}}$$

* When applying chain rules, we can define intermediate variables (nice and simple, can used as common results)
$$ 
    \delta = \frac{\partial L}{\partial s} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial s}
$$
$$
    \epsilon_j = \frac{\partial L}{\partial r_j} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial s} \frac{\partial s}{\partial u_j} \frac{\partial u_j}{\partial r_j}
$$
...  
We have
1. $\frac{\partial L}{\partial w_j} = \delta u_j = (z-y) u_j$
2. $\frac{\partial L}{\partial v_{ij}} = \epsilon_j x_i = \delta w_j g'(r_j) x_i$

---
#### Forward propagation (just compute outputs for each layer, make predictions)
* Use current estimates of $v_{ij}$ and $w_j \rightarrow$ Calculate $r_j, u_j, s$ and $z$
* 一次 forward 一次 backward

#### Further notes on ANN
* ANN's are flexible, but flipside is over-parameterisation, hence tendency to **overfitting**
* Starting weights usually random distributed about zero
* Implicit regularisation: **early stopping**
* Explicit regularisation
  * Much like ridge regression
  * With some activation functions this also shrinks the ANN towards a linear model


### Lecture 8: Deep learning, CNN, Autoencoders

#### Deep learning
* ANNs with a single hidden layer are **universal approximators**
  * E.g. OR, AND, NOT
* It's efficient to **stack** several hidden layers $\rightarrow$ Deep neural networks

#### Representation learning
* Consecutive layers form **representations of the input** of increasing complexity
* ANN using complex non-linear representation
* A hidden layer can be though of as the *transformed feature space* (e.g. $\mathbf{u} = \phi(\mathbf{x})$)
* Parameters of such a transformation are learned from data
* ANN layers as data transformation:
  
<img src="pic/transformation.png" width="400">

#### Depth v.s. width
* Width: single infinitely wide layer gives a universal approximator
* Depth: yields more accurate models
* Seek to mimic layered complexity in a network
* However **vanishing gradient problem** affects learning with very deep models

#### Convolutional Neural Networks
* In this example
  * C = 1, (2C+1) is the filter size
  * Stride = 1
<img src="pic/convolution.png" width="400">
* For 2D images
<img src="pic/conv2d.png" width="400">
* For computer vision
  * Use 5 filters for 2D convolution
  * Downsampling could be Max Pooling
  * Use 10 filters for 3D convolution
  
<img src="pic/cv.png" width="400">

#### Components of CNN
* Convolutional layers
  * Complex input representations based on convolution operation
  * Weights of the filters are learned from training data
* Downsampling
  * Re-scales to smaller resolution, imits parameter explosion
  * Usually via Max Pooling
* Fully connected parts and output layer
  * Merges representations together

#### Downsampling via max pooling
* Special type of processing layer. For $m \times m$ patch
$$ v = max(u_{11}, u_{12}, ..., u_{mm}) $$
* Strictly speaking, not everywhere differentiable (pooling layers not differentiable). Instead, gradient is defined according to **"sub-gradient"**
* Max pooling:
  * Tiny changes in values of $u_{ij}$ that is not max do not change $v$
  * If $u_{ij}$ is max value, tiny changes in that value change $v$ linearly
  * Use $\frac{\partial v}{\partial u_{ij}} = 1$ if $u_{ij} = v$, and $\frac{\partial v}{\partial u_{ij}}$ otherwise
* **Forward pass** records maximising element, which is then used in the backward pass during back-propagation

#### Convolution as a regulariser
<img src="pic/cv_regular.png" width="400">

#### Autoencoder
* Given data without labels $x_1, ..., x_n$
  * Set $y_i \equiv x_i$ (target/output = input)
  * train an ANN to predict $z(x_i) \approx x_i$ (approximate input)
* Set bottleneck layer (**representation**) $u$ in middle "thinner" than input
<img src="pic/autoencoder.png" width="400">

#### Bottleneck
* Manage to train a network that gives a good **restoration** of the original signal $z(x_i) \approx x_i$
* That means that the data structure can be effectively **described (encoded) by a lower dimensional representation $\mathbf{u}$**

#### Dimensionality reduction
* Autoencoders can used for **compression** and **dimensionality reduction** via a non-linear **transformation**
* If you use **linear activation functions** and **only one hidden layer**, then the setup becomes almost that of **PCA**

### Lecture 9: Support Vector Machine

#### Linear hard-margin SVM
* **Binary classifier**
  * $s = b + \sum_{i = 1}^m x_i w_i$
  * Predict class A if $s \geq 0$
  * Predict class B if $s < 0$
* **Linear classifier**
  * $s$ is a linear function of inputs, and the **separating boundary** is linear
* Model the data as **linearly separable**
  * There exists a hyperplane perfectly separating the classes
* Training using all data at once

#### SVM vs. Perceptron
* Given learned parameter value, an **SVM makes predictions exactly like a perceptron**
* Different ways to learn parameter
  * SVM: maximise margin
  * Perceptron: min perceptron loss

#### Separation boundary
* Choosing parameters means choosing a separating boundary (hyperplane)
* **For perceptron,** all boundaries that separates classes perfectly are equally good
  * since perceptron loss is 0 for all of them
* **For SVM,** it aim to find the separation boundary that maximises the margin between the classes
* **Margin** is defined by the **location(b) and orientation(w)** of the separating boundary, which are defined by SVM parameters
  * Space between two dashed lines

#### Margin width
* Distance between the hyperplane and the nearest data points
  * therefore, **distance to the nearest red and blue points are the same**
* Points on margin boundaries called **support vectors**
  
<img src="pic/svm.png" width="400">

#### SVM Parameters
* The separation boundary is defined by parameters $\mathbf{w}$ and $b$
  * $\mathbf{w}$: vetor normal (perpendicular) to the boundary
  * $b$: bias / intercept
* For a point X, let $X_p$ denote the **projection** of X onto the hyperplane
* Distance from point to hyperplane
  * $||r|| = \pm \frac{w'x + b}{||w||}$
* In training data, $y_i$ corresponds to binary label (-1 or 1)
  * $y_i$ encode the side of the boundary each $x_i$ is on
* Thus, **distance from i-th point** to a perfect hyperplane:
  * $||r_i|| = \frac{y_i(w'x_i + b)}{||w||}$

#### SVM: Finding separating boundary
* Margin width = distance from the hyperplane to the closest point
* **SVM Objective:** 
  * Maximise ($\min_{i = 1, ..., n} \frac{y_i(w'x_i + b)}{||w||}$) as a function of $\mathbf{w}$ and $b$
  * Problem: non-unique representation of separating boundary (hyperplane)
    * i.e. can use any $\alpha (w'x + b) = \tilde{w}'x + \tilde{b} = 0$
    * Infinite number of solutions
  * Possible solution to **resolve ambiguity**: 
  * Measure the distance to the closest point ($i^*$) and rescale param s.t. 
    * margin: $\frac{y_{i^*}(w'x_{i^*} + b)}{||w||} = \frac{1}{||w||}$
    * (Arbitrary set the numerator to 1 to get unique values of $\mathbf{w}$ and $b$)
* **SVM Objective with extra requirement:**
  * Extra requirement: Set margin width = $\frac{y_{i^*}(w'x_{i^*} + b)}{||w||} = \frac{1}{||w||}$
  * $i^*$ denotes index of closest example to boundary
* Therefore, (hard margin) SVM aims to find:
  * $\argmin_w ||w||$ s.t. $y_i (w'x_i + b) \geq 1$ for $i = 1, ..., n$
  * Minimum $||w||$ => maximise margin
  * Constraint: perfect separation of points

#### SVM as regularised ERM
* SVM objective
  * $\argmin_w ||w||$: data-independent regularisation term
  * $y_i (w'x_i + b) \geq 1$ for $i = 1, ..., n$: constraints as data-dependent training error
    * Can be interpreted as loss
      * $l_\infty = 0$ if prediction correct
      * $l_\infty = \infty$ if prediction wrong (give infinite loss to make perfect separation)

### Lecture 10: Soft-Margin SVM, Lagrangian Duality

#### Soft-Margin SVM
* One of the three approach to fit non-linearly separable data
  1. Transform the data (kernel)
  2. Relex the constraints (Soft-Margin)
  3. Combination of (1) and (2)
* Relax constraints to allow points to be:
  * **Inside the margin**
  * Or on the **wrong side of the boundary**
* Penalise boundaries by the **extent of "violation"** (distance from margin to wrong points)

#### Hinge loss: soft-margin SVM loss
* **Hard-margin SVM loss:**
  * $l_\infty = 0$ if prediction correct
  * $l_\infty = \infty$ if prediction wrong
* **Soft-margin SVM loss: (hinge loss)**
  * $l_h = 0$ if prediction correct
  * $l_h = 1 - y(w'x + b) = 1 - y\hat{y}$ if prediction wrong (penalty)
  * Can be written as: $l_h = max(0, 1 - y_i (w'x_i + b))$
* Compare with perceptron loss
  * $L(s,y)=max(0,−sy)$

#### Soft-Margin SVM Objective
* $\argmin_{\mathbf{w}, b} (\sum_{i = 1}^n l_h(x_i, y, w, b) + \lambda ||w||^2)$
  * Like ridge regression
* Reformulate objective:
  * Define **slack variables** as upper bound on loss
    * Allow you to relax the constraint
    * $\xi_i \geq l_h = \max(0, 1 - y_i (w'x_i + b))$
    * Non-zero means there is some violation
    * Don't like function like this in optimisation (no derivative)
  * **Then, new objective:**
    * $\argmin_{w,b,\xi} (\frac{1}{2} ||w||^2 + C \sum_{i = 1}^n \xi_i)$ 
    * Constraints:
      * $\xi_i \geq 1 - y_i (w'x_i + b)$ for $i = 1, ..., n$
      * $\xi_i \geq 0$ for $i = 1, ..., n$
      * (Penalise based on the size of $\xi_i$, like having loss function in objective)
      * **$\xi$ gets pushed down to be equal to $l_h$**
    * C: hyperparameter (have to tune by gridSearch)

#### Two variations of SVM
* **Hard-margin SVM objective:**
  * $\argmin_{w,b} \frac{1}{2}||w||^2$
  * s.t. $y_i(w'x_i + b) \geq 1$ for $i = 1, ..., n$
* **Soft-margin SVM objective:**
  * $\argmin_{w,b} \frac{1}{2}||w||^2$
  * s.t. $y_i(w'x_i + b) \geq 1 - \xi_i$ for $i = 1, ..., n$ and $\xi_i \geq 0$ for $i = 1, ..., n$ 
  * The constraints are **relaxed** by allowing violation by $\xi_i$

#### Constraint optimisation
* **Canonical form:**
  * minimise $f(x)$
  * s.t. $g_i(x) \leq 0, i = 1, ..., n$
  * and $h_j(x) = 0, j = 1, ..., m$
* Training SVM is also a constrained optimisation problem
* Method of **Lagrange multipliers**
  * Transform to unconstrained optimisation
  * (Or) Transform **primal** program to a related **dual** program
    * Analyze necessary & sufficient conditions for solutions of both program

#### Lagrangian and duality
* **Dual** objective function:
  * $L(x, \lambda, v) = f(x) + \sum_{i = 1}^n \lambda_i g_i(x) + \sum_{j = 1}^m v_j h_j(x)$
  * Primal constraints became penalties
  * Called **Lagrangian** function
  * New $\lambda$ and $v$ are called the **Lagrange multipliers** or **dual variables**
* Primal program: $\min_x \max_{\lambda \geq 0} L(x, \lambda, v)$
* Dual program: $\max_{\lambda \geq 0, v} \min_{x} L(x, \lambda, v)$
  * May be easier to solve, advantageous
* Duality
  * Weak duality: dual optimum $\leq$ primal optimum
  * For convex problem, we have strong duality: optima coincide (same optima for primal and dual)
    * Including SVM

#### Dual program for hard-margin SVM
* Minimise Lagrangian w.r.t to primal variables <=> maximise w.r.t dual variables yields the **dual program:**
  * $\argmax_\lambda \sum_{i = 1}^m \lambda_i - \frac{1}{2}\sum_{i = 1}^n \sum_{j = 1}^n \lambda_i \lambda_j y_i y_j x'_i x_j$
  * s.t. $\lambda_i \geq 0$ and $\sum_i^n \lambda_i y_i = 0$
* According to strong duality, solve dual <=> solve primal
* Complexity of solution:
  * $O(n^3)$ instead of $O(d^3)$
* Program depends on dot products of data only -> kenel

#### Making predictions with dual solution
* **Recovering primal variables**
  * From stationarity: get $w_j^*$
    * $w_j^* = \sum_{i = 1}^n \lambda_i y_i (x_i)_j = 0$
  * From dual solution (complementary slackness): (get $b^*$)
    * $y_j(b^* + \sum_{i = 1}^n \lambda_i^* y_i x'_i x_j) = 1$
    * For any example $j$ with $\lambda_i^* > 0$ **(support vectors)**
* Make predictions (testing)
  * Classify new instance $x$ based on sign of
    * $s = b^* + \sum_{i = 1}^n \lambda_i^* y_i x'_i x$
    * ($s = w'x + b$)

#### Optimisation for Soft-margin SVM
* Training: find $\lambda$ that solves (dual)
  * $\argmax_\lambda \sum_{i = 1}^m \lambda_i - \frac{1}{2}\sum_{i = 1}^n \sum_{j = 1}^n \lambda_i \lambda_j y_i y_j x'_i x_j$
  * s.t. $C \geq \lambda_i \geq 0$ (box constraints) and $\sum_i^n \lambda_i y_i = 0$
  * Where $C$ is a box constraints **(only difference between soft and hard SVM)**
    * Vector $\lambda$ is inside a box of side length $C$
    * Big $C$: penalise more training data, let training data has more influence
    * Small $C$: don't care about training data, want big margins
* Make predictions: (same as hard margin)
  * Classify new instance $x$ based on sign of
    * $s = b^* + \sum_{i = 1}^n \lambda_i^* y_i x'_i x$

#### Complementary slackness
* One of the KKT conditions:
  * $\lambda_i^* (y_i((w^*)'x_i + b^*) - 1) = 0$
* Remember:
  * $y_i (w'x_i + b) - 1 > 0$ means that $x_i$ is outside the margin (classified correctly)
* Points outside the margin must have $\lambda_i^* = 0$
* Points with non-zero $\lambda^*$ are **support vectors**
  * $w^* = \sum_{i = 1}^n \lambda_i y_i x_i$
  * Other points has no influence on $w^*$ (orientation of hyperplane)

#### Training SVM
* Inefficient
* Many $\lambda$s will be zero (sparsity)

### Lecture 11: Kernel Methods

#### Kernelising the SVM
* Two ways to handle non-linear data with SVM
  1. Soft-margin SVM
  2. Feature space transformation
     * Map data to a new feature space
     * Run hard-margin / soft-margin SVM in new feature space
     * Decision boundary is non-linear in original space
* Naive workflow
  1. Choose / design a linear model
  2. Choose / design a high-dimensional transformation $\phi(x)$
      * **Hoping** that after adding a lot of various features, some of them will make the daa linearly separable
  3. For each training example and each new instance, compute $\psi(x)$
* **Problem:** impractical / impossible to compute $\psi(x)$ for high / infinite-dimensional $\psi(x)$
* **Solution:** Use kernel function
  * Since both training and prediction process in SVM only depend **dot products** between data points
  * Before data transformation:
    * Training: (parameter estimation)
      * $\argmax_\lambda \Sigma_{i=1}^n - \frac{1}{2}\Sigma_{i=1}^n \Sigma_{j=1}^n \lambda_i \lambda_j y_i y_j \mathbf{x_i'x_j}$
    * Making prediction: (computing predictions)
      * $s = b^* + \Sigma_{i=1}^n \lambda_i^* y_i \mathbf{x_i'x}$
  * After data transformation:
    * Training: 
      * $\argmax_\lambda \Sigma_{i=1}^n - \frac{1}{2}\Sigma_{i=1}^n \Sigma_{j=1}^n \lambda_i \lambda_j y_i y_j \mathbf{\phi(x_i)'\phi(x_j)}$
    * Making predictions:
      * $s = b^* + \Sigma_{i=1}^n \lambda_i^* y_i \mathbf{\phi(x_i)' \phi(x)}$

#### Kernel representation
* Kernel:
  * A function that can be expressed as a dot product in some feature space: $K(u,v) = \psi(u)' \psi(v)$
* For some $\psi(x)$'s, kernel is faster to compute directly than first mapping to feature space then taking dot product
  * A "shortcut" function that gives exactly the same answer $K(x_i, x_j) = k_{ij}$
* Then, SVM becomes:
  * Training: 
      * $\argmax_\lambda \Sigma_{i=1}^n - \frac{1}{2}\Sigma_{i=1}^n \Sigma_{j=1}^n \lambda_i \lambda_j y_i y_j \mathbf{K(x_i, x_j)}$
  * Making predictions:
    * $s = b^* + \Sigma_{i=1}^n \lambda_i^* y_i \mathbf{K(x_i, x)}$

#### Approaches to non-linearity
* ANNs:
  * Elements of $u = \phi(x)$ (second layer neurons) are transformed input $x$
  * This $\phi$ has weights learned from data
* SVMs:
  * Choice of kernel $K$ determines feature $\phi$
  * Don't learn $\phi$ weights
  * But, don't even need to compute $\phi$ so can support very high dimensional $\phi$
  * Also support arbitrary data types

#### Modular learning
* All information about feature mapping is **concentrated within the kernel**
* In order to use a different feature mapping -> change the kernel function
* Algorithm design decouples into:
  1. Choosing a "learning method" (e.g. SVM v.s. logistic regression)
  2. Choosing feature space mapping (i.e. kernel)
* Representer theorem
  * For any training set ${x_i, y_i}_{i=1}^n$, any empirical risk function $E$, monotonic increasing function $g$, then any solution: 
    * $f^* \argmin_f E(x_1, y_1, f(x_1), ..., x_n, y_n, f(x_n)) + g(||f||)$ 
    * has representation for some coefficients: $f^* (x) = \Sigma_{i=1}^n \alpha_i k(x, x_i)$
  * Tells us when a learner is kernelizable
  * The dual tells us the form this linear kernel representation takes
  * (SVM is an example)

#### Constructing kernels
* Polynomial kernel
  * $K(u,v) = (u'v + c)^d$
  * Can add $\sqrt{c}$ as a dummy feature to $u$ and $v$ (dim + 1)
  * $(u'v)^d = (u_1 v_1 + ... + u_m v_m)^d = \Sigma_{i=1}^l (u_1 v_1)^{a_{i1}}...(u_m v_m)^{a_{im}} = \Sigma_{i=1}^l (u_1^{a_{i1}} ... u_m^{a_{im}}) = \Sigma_{i=1}^l \phi(u)_i \phi(v)_i$
  * Feature map $\phi$: $\R^m \rightarrow \R^l$, where $\phi_i(x) = (x_1^{a_{i1}}, ..., x_m^{a_{im}})$
* Identifying new kernels
  1. Method 1: Let $K_1(u,v), K_2(u,v)$ be kernels, $c > 0$ be a constant, and $f(x)$ be a real-valued function, then each of the following is also a kernel:
       * $K(u,v) = K_1(u,v) + K_2(u,v)$
       * $K(u,v) = cK_1(u,v)$
       * $K(u,v) = f(u)K_1(u,v)f(v)$
  2. Method 2: Use Mercer's theorem
       * Consider a finite sequences of objects $x_1, ..., x_n$
       * Construct $n \times n$ matrix of pairwise values $K(x_i, x_j)$
       * $K(x_i, x_j)$ is a **valid kernel** if this matrix is **positive semi-definite (PSD)**, and this holds for all possible sequence $x_1, ..., x_n$
* Remember we need $K(u,v)$ to imply a dot product in some feature space

### Lecture 12: Ensemble methods

#### Combining models (Ensembling)
* Construct a set of base models (learners) from given training set and aggregates the outputs into a single meta-model (ensemble)
  * Classification via (weighted) majority vote
  * Regression vis (weighted) averaging
  * More generally: meta-model = $f($base model$)$
* Recall bias-variance trade-off:
  * $E[l(y, \hat{f}(x_0))] = (E[y] - E[\hat{f}])^2 + Var[\hat{f}] + Var[y]$
  * Averaging $k$ independent and identically distributed predictions **reduces variance**: $Var[\hat{f}_{avg}] = \frac{1}{k} Var[\hat{f}]$
* Three methods:
  * Bagging and random forests
  * Boosting
  * Stacking

#### Bagging (bootstrap aggregating)
* Method: construct "near-indendent" datasets via **sampling with replacement**
  * Generate $k$ datasets, each size $n$
  * Build base classifiers on each constructed dataset
  * Aggregate predictions via voting / averaging
* Bagging example: Random Forest
  * Select random subset of $l$ of the $m$ features
  * Train decision tree on bootstrap sample using the $l$ feature
  * Works extremely well in many practical settings
* Reflections
  * Simple method based on sampling and voting
  * Possibility to parallelise computation of individual base classifiers
  * Highly effective over noisy datasets
  * Performance is often significantly better than (simple) base classifiers, never substantially worse
  * Improve unstable classifiers by reducing variance

#### Using out-of-sample data
* For each round, a particular sample has probability of $(1 - \frac{1}{n})$ of not being selected
  * Probability of being left out is $(1 - \frac{1}{n})^n$
  * For large $n$, $e^{-1} \approx 0.368$
  * On average, only $63.2\%$ of data included per bootstrap sample
* Can use this for independent error estimate of ensemble
  * OOB (Out-Of-Bag) Error
  * Safe like CV, but on overlapping sub0samples
  * Evaluate each base classifier on its out-of-sample $36.8\%$
  * Average these evaluation $\rightarrow$ Evaluation of ensemble

#### Boosting
* Intuition: 
  * Focus attention of base classifiers on examples "hard to classify"
* Method: iteratively change the distribution on examples to reflect performance of the classifier on the previous iteration
  * Start with each training instance having $1/n$ probability of being included in the sample
  * Over $k$ iterations, train a classifier and **update the weight of each instance** according to classifier's ability to classify it
    * Misclassified -> give more weight to that instance
  * Combine the base classifiers via **weighted voting**

#### Adaboost
1. Initialise example distribution $P_1(i) = 1 / n$
2. For $c = 1 ... k$
   1. Train base classifier $A_c$ on sample with replacement from $P_c$
   2. Set (classifier) confidence $\alpha_c = \frac{1}{2}\ln(\frac{1 - \epsilon_c}{\epsilon_c})$ for classifier's error rate $\epsilon_c$
   3. Update example distribution to be normalised of:
        * $P_{c+1}(i) \propto P_c(i) \times \exp(-\alpha_c)$ if $A_c(i) = y_i$ (correct prediction)
        * $P_{c+1}(i) \propto P_c(i) \times \exp(\alpha_c)$ if otherwise (wrong prediction)
   4. Classify as majority vote weighted by confidences $\argmax_y \Sigma_{c=1}^k \alpha_t \delta(A_c(x) = y)$
* Technically: Reinitialise example distribution whenever $\epsilon_c > 0.5$
* Base learners: often decision stumps or trees, anything "weak"
* Reflections
  * Method based on **iterative sampling** and **weighted voting**
  * More computationally expansive than bagging
  * The method has **guaranteed performance** in the form of error bounds over the training data
  * In practical applications, boosting can overfit
    * (Can do hybrid of bagging and boosting, but if using too many base classifiers -> overfit)

#### Bagging v.s. Boosting
* Bagging
  * Parallel sampling
  * Minimise variance
  * Simple voting
  * Classification or regression
  * Not prone to overfitting
* Boosting
  * Iterative sampling
  * Target "hard" instances
  * Weighted voting
  * Classification or regression
  * Prone to overfitting (unless base learners are simple)

#### Stacking
* Intuition: "smooth" errors over a range of algorithms with different biases
* Method: train a meta-model over the outputs of the base learners
  * Train base- and meta-learners using CV
  * Simple meta-classifier: logistic regression
* Generalisation of bagging and boosting
* Reflections:
  * Compare this to ANNs and basis expansion
    * Mathematically simple but computationally expansive method
    * Able to combine heterogeneous classifiers with varying performance
    * With care, stacking results in as good or better results **than the best of the base classifier**

### Lecture 13: Multi-armed bandits

#### Stochastic multi-armed bandits
* Learn to take actions
  * Receive only indirect supervision in the form of **rewards**
  * Only observe rewards for actions taken
  * Simplest setting with an **explore-exploit trade-off**

#### Exploration v.s. Exploitation
* "Multi-armed" bandit (MAB)
  * Simplest setting for balancing **exploration, exploitation**
  * Same family of ML tasks as reinforcement learning
* Numerous applications
  * Online advertising
  * Stochastic search in games
  * Adaptive A/B testing
  * ...

#### Stochastic MAB setting
* Possible actions $\{1, ..., k\}$ called **"arms"**
  * Arm $i$ has distribution $P_i$ on bounded rewards with mean $\mu_i$
* In round $t = 1 .. T$
  * Play action $i_t \in \{ 1, ..., k\}$ (possibly randomly)
  * Receive reward $X_{i_t}(t) \sim P_{i_t}$
* Goal: miniise cumulative **regret**
  * $\mu^*T - \Sigma_{t=1}^T E[X_{i_t}(t)]$
  * Where $\mu^* = \max_i \mu_i$

#### Greedy
* At round $t$
  * Estimate value of each arm $i$ as **average reward** observed
    * $Q_{t-1}(i) = \frac{\Sigma_{s=1}^{t-1} X_i(s) I(i_s = i)}{\Sigma_{s=1}^{t-1} I(i_s = i)}$, if $\Sigma_{s=1}^{t-1} I[i_s = i] > 0$
    * $Q_{t-1}(i) = Q_0$, otherwise
    * Init constant: $Q_0(i) = Q_0$ used until arm $i$ has been pulled
  * Exploit
    * $i_t \in \argmax_{i \leq i \leq k} Q_{t-1}(i)$
  * Tie breaking randomly

#### $\epsilon$-Greedy
* At round $t$
  * Estimate value of each arm $i$ as average reward observed
    * $Q_{t-1}(i) = \frac{\Sigma_{s=1}^{t-1} X_i(s) I(i_s = i)}{\Sigma_{s=1}^{t-1} I(i_s = i)}$, if $\Sigma_{s=1}^{t-1} I[i_s = i] > 0$
    * $Q_{t-1}(i) = Q_0$, otherwise
    * Init constant: $Q_0(i) = Q_0$ used until arm $i$ has been pulled
  * Exploit
    * $i_t \in \argmax_{i \leq i \leq k} Q_{t-1}(i)$ w.p. $1 - \epsilon$
    * $i_t \in$ Unif($\{1, ..., k\}$) w.p. $\epsilon$
  * Tie breaking randomly
* Hyperparameter $\epsilon$ controls exploration v.s. exploitation
* Does better long-term (than Greedy) by exploring
* Pessimism v.s. Optimism:
  * Pessimism: Init Q's below observable reward -> Only try one arm (E.g. $Q_0 = -10$)
  * Optimism: Init Q's above observable rewards -> Explore arms at least once (E.g. $Q_0 = 10$)
  * Middle-ground init Q -> Explore arms at most once
  * Pure greedy never **explores** an arm more than once
* Limitations:
  * Exploration and exploitation are too distinct
    * Exploration actions completely blind to promising arms
    * Initialisation tricks only help with "cold start"
  * Exploitation is blind to **confidence** of estimates

#### Upper Confidence Bound (UCB)
* At round $t$
  * Estimate value of each arm $i$ as average reward observed
    * $Q_{t-1}(i) = \hat{\mu}_{t-1}(i) + \sqrt{\frac{2\log(t)}{N_{t-1}(i)}}$, if $\Sigma_{s=1}^{t-1} I[i_s = i] > 0$
    * $Q_{t-1}(i) = Q_0$, otherwise
    * Init constant: $Q_0(i) = Q_0$ used until arm $i$ has been pulled
  * $\hat{\mu}_{t-1}(i) = \frac{\Sigma_{s=1}^{t-1} X_i(s) I(i_s = i)}{\Sigma_{s=1}^{t-1} I(i_s = i)}$
  * $N_{t-1}(i) = \Sigma_{s=1}^{t-1} I[i_s = i]$
  * Exploit
    * $i_t \in \argmax_{i \leq i \leq k} Q_{t-1}(i)$
  * Tie breaking randomly
  * (upper bound for **explore** boost)
  * Addresses several limitation of $\epsilon$-Greedy
  * Can "pause" in a bad arm for a while, but eventually find best
  * Results:
    * Quickly overtakes the $\epsilon$-Greedy approaches
    * Continues to outspace on per round rewards for some time
    * More striking when viewed as mean cumulative rewards
  * Notes:
    * Theoretical **regret bounds**, optimal up to multiplicative constant
    * Tunable $\rho > 0$ exploration hyperparam, can replace "2"
    * Captures different $\epsilon$ rates & bounded rewards outside [0,1]
    * Many variations e.g. different confidence bounds

### Lecture 14: Bayesian regression

#### Bayesian Inference
* Idea
  * Weights with a better fit to the training data should be more probable than others
  * Make predictions with all these weightsm scaled by their probability
* Reason under all possible parameter values
  * weighted by their posterior probability
* More robust predictions
  * less sensitive to overfitting, particularly with small training sets
  * Can give rise to more expensive model class

#### Frequentist v.s. Bayesian
* Frequentist: **learning using point estimates**, regularisation, p-values
  * backed by sophisticated theory in simplifying assumptions
  * mostly simpler algorithms, characterises much practical machine learning research
* Bayesian: maintain **uncertainty**, marginalise out unknowns during inference
  * some theory
  * often more complex algorithms, but not always
  * often more computationally expensive

#### Bayesian Regression
* Application of bayesian inference to linear regression, using normal prior over $w$
* Consider full posterior $p(w | X, y, \sigma^2)$
* Sequential Bayesian updating
  * Can formula $p(w | X, y, \sigma^2)$ for given dataset
  * As we see more and more data:
    1. Start with prior $p(w)$
    2. See new labelled datapoint
    3. Compute posterior $p(w | X, y, \sigma^2)$
    4. The **posterior now takes role of prior** & repeat from step 2

#### Conjugate Prior
* Product of **likelihood $\times$ prior**: results in the same distribution as the prior

#### Stages of Training
1. Decide on model formulation & prior
2. Compute **posterior** over parameters $p(w | x,y)$
3. 3 methods:
   1. MAP:
      1. Find mode for $w$
      2. Use to make prediction on test
   2. Approx. Bayes:
      1. Sample many $w$
      2. Use to make ensemble average prediction on test
   3. Exactly Bayes
      1. Use all $w$ to make expected prediction on test

#### Prediction with uncertain $w$
* Could predict using sampled regression curves
  * Sample $S$ parameters, $w^{(s)}, s \in \{ 1, ..., S\}$
  * For each sample, compute prediction $y_*^{(s)}$ at test point $x_*$
  * (Monte Carlo integration)
* For Bayesian regression, there's a simpler solution:
  * Integration can be done analytically, for
  * $p(\hat{y}_* | X, y, x_*, \sigma^2) = \int p(w | X, y, \sigma^2) p(y_*| x_*, w, \sigma^2) dw$
* Pleasant properties of Gaussian distribution means integration is tractable
  * $p(\hat{y}_* | X, y, x_*, \sigma^2) = ... = \text{Normal}(y_* | x_*' w_N, \sigma^2_N(x_*))$
  * $\sigma_N^2 = \sigma^2 + x_*'V_N x_*$
  * Additive variance based on x_* match to training data

#### Caveats (Notes)
* Assumption
  * known data noise parameter $
  * sigma^2$
  * data was drawn from the model distribution

### Lecture 15: Bayesian classification

#### Discrete Conjugate prior
* Example:
  * Prior: Beta
  * Likelihood: Binomial
  * Posterior: Beta (conjugacy)

#### Suite of useful conjugate priors
* Regression:
  1. For mean:
      * Likelihood: Normal
      * Prior: Normal
  2. For variance / covariance: 
      * Likelihood: Normal
      * Prior: Inverse Gamma / Inverse Wishart
* Classification:
  1. Likelihood: Binomial, Prior: Beta
  2. Likelihood: Multinomial, Prior: Dirichlet
* Counts:
  1. Likelihood: Poisson, Prior: Gamma

#### Bayesian Logistic Regression
* Discriminative classifier which conditions on inputs
* Similar problems with parameter uncertainty compared to regression
* Need prior over $w$ (coefficients), not $q$
* **No known conjugacy**
  * Thus, use a Gaussian prior
* Resolve by (Laplace) approximiation:
  * Assume posterior $\approx$ Normal about mode
  * Can compute normalisation constant, draw samples, etc.

### Lecture 16: PGM Representation

#### PGM
* Mariage of graph theory and probability theory
* Tool of choice for Bayesian statistical learning

#### Bayesian statistical learning v.s. PGM (aka. "Bayes Nets")
* Bayesian Statistical learning
  * Model joint distribution of $X$'s, $Y$, and parameters $r.v.$'s
  * Priors: marginals on parameters
  * Training:
    * update prior to posterior using observed data
  * Prediction:
    * output posterior, or some function of it (MAP)
* PGM ("Bayes Nets")
  * Efficient joint representation
    * Independence made explicit
    * Trade-off between expressiveness and need for data, easy to make
    * Easy for Practitioners to model
  * Algorithms to fit parameters, compute marginals, posterior

#### Joint distribution
* All joint distributions on discrete $r.v.$'s can be represented as table
  * Table assign probability per row
* We can make probabilistic inference from joint on $r.v.$'s
  * Compute any other distributions involving our $r.v.$'s
  * **Bayes rule + marginalisation**
  * Example: Naive Bayes
* Bad: Computational complexity
  * Tables have exponential number of rows in number of $r.v.$'s
  * Therefore -> poor space & time to marginalise
* Ugly: Model complexity
  * Way too flexible
  * Way too many parameters to fit
    * Need lots of data OR will overfit

#### Independence
* If assume S, T independent, model need 6 params
  * $P(S,T)$ factors to P(S), P(T) -> 2 params
  * $P(L|T,S)$ modelled in full -> 4 params
* For assumption-free model, need 7 params
  * $P(L,T,S)$ modelled in full -> $2^3-1 = 7$ params
* Independence assumptions
  * Can be reasonable in light of domain expertise
  * Allow us to **factor** -> Key to tractable models

#### Factoring Joint Distributions
* Chain Rule: For **any ordering** of $r.v.'s$ can always factor:
  * $P(X_1, X_2, ..., X_k) = \prod_{i=1}^k P(X_i | X_{i+1}, ..., X_k)$
* Model's independence assumptions correspond to:
  * Dropping conditioning $r.v.$'s in the factors
  * E.g. Unconditional independence: $P(X_1 | X_2) = P(X_1)$
  * E.g. Conditional independence: $P(X_1 | X_2, X_3) = P(X_1 | X_2)$ 
    * Given $X_2$, $X_1$ and $X_3$ independent
* Simpler factors: speed up inference and avoid overfitting

#### Directed PGM
* Nodes -> Random variables
* Edges -> Conditional independence
  * Node table: $P(child|parents)$
  * Child **directly** depends on parents
* **Joint factorisation**
  * $P(X_1, ..., X_k) = \prod_{i=1}^k P(X_i | X_j \in parents(X_i))$

<img src="pic/pgm.png" width="200">

#### Plate notation
* Short-hand for repeats
* Simplifying growing complicated PGM

<img src="pic/plate.png" width="400">

#### PGM: frequentist v.s. Bayesian
* PGM -> joints
* Bayesian add: node per param

<img src="pic/bayes.png" width="400">

#### Undirected PGMs
* Parameterised by **arbitrary positive valued functions** of the variables and **global normalisation**
  * Aka. Markov Random Field
* Undirected v.s. Directed PGM
  * Undirected:
    * Graph with undirected edges
    * Probability:
      * Each node a $r.v.$
      * Each **clique $\mathbf{C}$ has "factor"**: 
        * $\psi_C(X_j:j \in C) \geq 0$
      * Joint $\propto$ product of factors
  * Directed:
    * Graph with directed edges
    * Probability:
      * Each node a $r.v.$
      * Each **node has conditional probability**:
        * $p(X_i | X_j \in parents(X_i))$
      * Joint = product of conditional probabilities
  * Key difference = **normalisation**
    * $\propto$ in undirected PGM

#### Undirected PGM formulation
* Based on notion of:
  * Clique: a set of **fully connecte**d nodes
  * Maximal clique: largest cliques in graph
* Joint probability defined as:
  * (Product of all the cliques)
  * $P(a,b,c,d,e,f) = \frac{1}{Z} \psi_1(a,b) \psi_2(b,c) \psi_3(a,d) \psi_4(d,c,f) \psi_5(d,e)$
  * where $\psi$ is a **positive function**
  * and $Z$ is the **normalising** "partition" function
    * $Z = \sum_{a,b,c,d,e,f} \psi_1(a,b) \psi_2(b,c) \psi_3(a,d) \psi_4(d,c,f) \psi_5(d,e)$

<img src="pic/undirected.png" width="400">

#### Directed to undirected
* Directed PGM formulated as:
  * $P(X_1, ..., X_k) = \prod_{i=1}^{k} P(X_i | X_{\pi_i})$
  * where $\pi$ indexes parents
* Equivalent to U-PGM with
  * each **conditional probability** term is **included** in one factor function, $\psi_c$:
    * clique structure links **groups of variables**
    * normalisation term trivial, Z = 1
* Turning D-PGM to U-PGM:

<img src="pic/u-pgm.png" width="400">

#### Why U-PGM
* Pros:
  * **Generalisation** of D-PGM
  * Simpler means of modelling without the need for per-factor normalisation
  * General inference algorithms use U-PGM representation
    * (Support both types of PGM)
* Cons:
  * (Slightly) weaker independence
  * Calculating global normalisation term (Z) intractable in general

#### PGM examples
* Hidden Markov Model (HMM)
  * **Directed**
  * Sequential observed outputs from hidden states
  * States: ejections & transitions
  * 2 assumptions:
    * Markov assumption
    * Output independence assumption
  * Applications:
    * NLP
    * Speech recognition
    * Biological sequences
    * Computer vision
  * Fundamental tasks (corresponding):
    * HMM: 
      * Evaluation: determine likelihood $P(O | \mu)$
        * $O$: observation sequence
        * $\mu$: HMM
      * Decoding: determine most probable hidden state $Q$
      * Learning: learn parameters $A, B, \Pi$
    * PGM:
      * Probabilistic inference
      * MAP point estimate
      * Statistical inference
* Kalman filter
  * Same with continuous Gaussian $r.v.$'s
* Conditional Random Field (CRF)
  * **Undirected**
  * Same model applied to sequences
    * Observed outputs are words, speech, etc.
    * States are tags: part-of-speech, alignment, etc.
  * Discriminative: model $P(Q | O)$
    * v.s. HMM's which are generative $P(Q, O)$
    * undirected PGM more general and expressive

<img src="pic/crf.png" width="400">

### Lecture 17: PGM Probabilistic and Statistical Inference

#### Probabilistic inference on PGMs
* Computing marginal and conditional distributions from the joint of a PGM 
  * Using **Bayes rule and marginalisation**
* Joint + Bayes rule + marginalisation -> anything !
* Example:
  * $P(HT | AS = t) = \frac{P(HT,AS=t)}{P(AS=t)} = \frac{\Sigma_{FG, HG, FA} P(AS=t, FA, HG, FG, HT)}{\Sigma_{FG, HG, FA, HT'} P(AS=t, FA, HG, FG, HT')}$
  * $HT'$ means All values of HT
  * Numerator: $\Sigma_{FG, HG, FA} P(HT) P(HG | HT, FG) P(FG) P(AS=t | FA, HG) P(FA)$
    * Can distribute the sums as far down as possible: $P(HT)\Sigma_{FG} P(FG) \Sigma_HG P(HG|HT, FG) \Sigma_FA P(FA) P(AS = t | FA, HG)$

#### Elimination algorithm
<img src="pic/elimination.png" width="400">
<img src="pic/ea.png" width="400">

#### Statistical inference on PGMs
* Learning (tables / params) from data
  * Fitting probability tables to observations
  * E.g. as a frequentist; a Bayesian would just use probabilistic inference to updat prior to posterior
* Probabilistic inference
  * **Computing other distributions** from joint
  * Elimination, sampling algorithms
* Statistical inference
  * Learn **parameters** from data
1. Fully-observed case (easy)
   * MLE -> maximise full joint (likelihood)
     * $\argmax_{\theta \in \Theta} \prod_{i=1}^n \prod_j p(X^j = x_i^j | X^{parents(j)} = x_i^{parents(j)})$
   * Decomposes easily, leads to **counts-based estimates**
     * Maximise log-likelihood instead; become sum of logs
     * Big maximisation of all params together, **decouples into small independent problems**
    * Example: 
      * Training a naive bayes classifier
      * (Counting -> probabilities)
2. Presense of **unobserved** variables
   * Latent, or unobserved variables
   * MLE:
     * Maximise likelihood of observed data only
     * Marginalise full joint to get desired "partial" joint
       * This won't decouple
   * Solution: Use **EM algorithm** !

### Lecture 18: Gaussian Mixture Model, Expectation Maximization

#### Unsupervised learning
* Aim: explore the structure (pattern, regularities) of the data
* Tasks:
  * Clustering (e.g. GMM)
  * Dimensionality Reduction (e.g. PCA)
  * Learning parameters of probabilistic models
* Applications:
  * Market basket analysis
  * Outlier detection
  * Unsupervised tasks in (supervised) ML pipelines

#### Gaussian Mixture Model (GMM)
* A probabilistic view of clustering
* Requires the user to choose the number of clusters in advance
* Gives a power to express **uncertainty about the origin** ("weights") of each point
  * Each point originates from cluster $c$ with probability $w_c$, $c = 1, ..., k$
* Still originats from one particular cluster, but not sure from which one
* Data points are samples from **a mixture of K distributions (components)**
  * In principle, we can adopt any probability distribution for the components
  * Howeer, normal distribution is a common modelling choice -> GMM
* (d-dimensional) Gaussian distribution
  * $N(x | \mu, \Sigma) = (2\pi)^{-\frac{d}{2}} |\Sigma|^{-\frac{1}{2}} \exp(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu))$
* Gaussian Mixture Distribution: (for each data point)
  * $P(x) = \Sigma_{j=1}^k w_j N(x | \mu_j, \Sigma_j) = \Sigma_{j=1}^k P(C_j) P(x|C_j)$
  * $P(x | C_j)$ is **conditional density** for component $j$
  * Here $P(C_j) \geq 0$ and $\Sigma_{j=1}^k P(C_j) = 1$
  * (Unkown) params of the model are:
    * $P(C_j), \mu_j, \Sigma_j$ for $j = 1, ..., k$
  
  <img src="pic/gmm.png" width="400">

#### Clustering as model estimation
* Clustering now amounts to **finding parameters** of the GMM that "best explain" the observed data
  * -> **MLE** that maximise $p(x_1, ..., x_n)$

#### Fitting the GMM
* Aim: find parameters that maximise $p(x_1, ..., x_n)$
* Cannot solve analytically by taking (first) derivative!
  * Solution: Use **Expectation-Maximisation (EM)**

#### Expectation Maximisation (EM) algorithm
* Motivation:
    1. Sometimes we **don't observe** some of the variables needed to compute the log-likelihood
        * Hidden variables
    2. Sometimes the form of log-likelihood is inconvenient to work with
        * (No closed form solution...)
 * EM is an algorithm
   * A way to **solve the problem posed by MLE**
   * Especially **convenient** under unobserved latent variables
 * MLE can be found by other methods
   * E.g. Gradient Descent

#### Algorithm (Simple version)
1. Initialisation $K$ clusters and their parameters
2. Iteration Step:
   * E-step:
     * Estimate the cluster of each data point
   * M-step:
     * Re-estimate the cluster parameters
       * $(\mu_j, \Sigma_j), p(C_j)$ for each cluster $j$

#### EM for GMM and generally
* EM is a general tool
  * Purpose: implement MLE under latent (missing) variables Z
* Variables and params in GMM
  * Variables: 
    * Observed: Point locations $X$
    * Hidden: Cluster assignments $Z$
  * Parameters: 
    * $\theta$: cluster locations and scales (and $p(C_j)'s$)
* What EM is really doing:
  * Coordinate ascent on **lower-bound on the log-likelihood**
    * M-step: ascent in modelled parameter $\theta$
    * E-step: ascent in marginal likelihood $P(Z)$
    * Each step moves towards a local optimum
    * Can get stuck (at local optima)
      * Need random restarts

#### Not-Examinable Part
* Log is a convex function (can use Jensen's inequality)
* Maximise $\log p(X|\theta)$ difficult
  * Maximise $\log p(X, Z|\theta)$ (log complete likelihood) instead
* Marginalise and use Jensen's Inequality to get lower-bound $E_Z [\log \frac{p(x,z|\theta)}{p(z)}]$
  * Note: $p(z) = p(z | x, \theta)$
* Maximising lowerbound:
  * **Equivalent** to maximising original incomplete likelihood
  * Since $p(z) = p(z | x, \theta)$ makes the lower bound tight
* Resulting EM algorithm:
  1. Initialisation: choose random initial values of $\theta^{(1)}$
  2. Update:
      * E-step: compute $Q(\theta, \theta^{(t)}) = E_{Z|X, \theta^{(t)}} [\log p(X, Z | \theta)]$
  3. Termination: If no change then stop
  4. Go to step 2
* The algorithm could results in local maximum

#### Estimating Parameters of GMM
* Can't compute the complete likelihood because we don't know z
* EM handles this by replacing $\log p(X,z|\theta)$ with $E_{Z|X, \theta^{(t)} [\log p(X,z|\theta)]}$
  * Requires: $p(z|X,\theta^{(t)})$
  * Assuming $z_i$ are pairwise independent, we need $P(z_i = c | x_i, \theta^{(t)})$
* E-step: calculating **cluster responsibility** (weights)
  * Use Bayes rule: $r_{ic} = P(z_i = c | x_i, \theta^{(t)}) = \frac{w_c N(x_i | \mu_c, \Sigma_c)}{\Sigma_{l=1}^k w_l N(x_i | \mu_l, \Sigma_l)}$
  * That (posterior) probability: responsibility that cluster $c$ takes for data point $i$
* M-step:
  * Take partial derivatives of $Q(\theta, \theta^{(t)})$ with respect to each of the parameters and set the derivatives to 0
  * Obtain new parameter estimates:
    <img src="pic/maximisation.png" width="400">
  * (Estimates for step (t+1))

#### K-means as a EM for a restricted GMM
* Consider GMM in which:
  * All components have the same fixed probability: $w_c = 1/k$
  * Each Gaussian has the fixed covariance matrix $\Sigma_c = \sigma^2 I$
  * **Only component centroids $\mathbf{\mu_c}$ need to be estimated**
* Approximate cluster responsibility:
  * Deterministic assignment: 
    * $r_ic = 1$ if centroid $\mu_c^{(t))}$ is closest to point $x_i$
    * $r_ic = 0$ otherwise
* Results in E-step: $\mu_c$ should be set as a centroid of points assigned to cluster $c$
* => k-means algorithm

### Lecture 19: Dimensionality Reduction

#### Dimensionality reduction
* Representing data using **a smaller number of variables** while preserving the "interesting" structure of the data
* Purposes:
  * Visualisation
  * Computational efficiency in a pipeline
  * Data compression or statistical efficiency in a pipeline
* Results in loss of information in general
  * Trick: ensure that most of the "interesting" information (signal) is preserved (while what is lost is mostly noise)

#### Pricinpal component analysis (PCA)
* Popular method for dimensionality reduction and data analysis
* Aim: find **a new coordinate system** s.t. most of the **variance is concentrated** along the first coordinate, then most of the remaining variance along the second coordinate, etc.
* Dimensionality reduction is based on **discarding coordinates** except the first $l < m$
  
#### Formulating the problem
* Projection of u on v: $u_v = u \cdot v / ||v||$
  * If $||v|| = 1$, $u_v = u \cdot v$
* Vector $v$ can be considered as a candicate **coordinate axis**
  * and $u_v$ the coordinate of point $u$ (on the new coordinate axis)
* Data transformation
  * Projecting all data points to a new coordinate axis $p_1$
    * result: $X'p_i$
    * Where $||p_i|| = 1$
    * $X$ has original data points in columns
* Sample covariance matrix:
  * For centered (mean subtracted) matrix $X$
  * $\Sigma_X = \frac{1}{n-1}X'X$

#### PCA
* Objective:
  * assume the data is centered
  * find $p_1$ to maximise variance along this PC: $p_1' \Sigma_x p_1$ subject to ||p_1|| = 1
* Constrained problem -> Use Lagrange multipliers
* Solution: $p_1$ is the eigenvector with corresponding $\lambda_1$ being the **max** eigenvalue (of covariance matrix $\Sigma_X$)
  * Variance captured by PC1: $\lambda_1 = p_1' \Sigma_X p_1$
* (Spectrum of a matrix is a set of its eigenvalues)
* Choose dimensions to keep from "knee" in scree plot

#### Efficient algorithm for PCA
* Setting $p_i$ as all eigenvectors of the **centered** data covariance matrix $\Sigma_X$ in decreasing eigenvalue order
* Lemma: a real symmetric m x m matrix has m real eigenvalues and corresponding eigenvectors are orthogonal
* Lemma: a PSD matrix further has non-negative eigenvalues

#### Linear regression v.s. PCA
* Another view of PCA: $s-dim$ plane minimising residual sum squares to data
* It turns out: 
  * PCA chooses the **direction** to be a hyperplane that minimise these errors (RSS)
  * Since variance and squared distance have something in common (both sum of squares)

<img src="pic/pca.png" width="400">

#### Additional effect of PCA
* Consider candidate axes $i$ and $(i+1)$, if there is correlation between them
  * This means that axis $i$ can be rotated further to capture more variance
* PCA should end up finding new axes (transformation) s.t. the transformed data is uncorrelated

#### Non-linear data and kernel PCA
* Low dimensional approximation need not be linear
* Kernel PCA: **map** data to feature space, **then** run **PCA**
  * Express PC in terms of data points
  * Solution uses X'X that can be kernelised:
    * $(X'X)_{ij} = K(x_i, x_j)$
  * Solution strategy differs from regular PCA
  * Modular: Changing kernel leads to a different feature space transformation