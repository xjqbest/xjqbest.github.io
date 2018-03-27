---
layout: post
title:  "Machine Learning Optimization"
date:   2016-12-24 14:51:00
categories: MachineLearning
tags: MachineLearning
excerpt: Machine Learning Optimization
---

# Introduction

## What is optimization  
找到在一组约束下的一个函数的极小值点:  
\begin{align}
\mathop{minimize\ }\limits_{x}f_0(x) \\\
s.t. \  f_i(x) &\le 0, i = \{1,...k\} \\\
h_j(x) &= 0, j = {1,...l}
\end{align}

## Why we care
优化是许多机器学习算法的核心。例如：   
1. Linear Regression  
\begin{align}
\mathop{minimize\ }\limits_{w}{|| Xw - Y ||}^2
\end{align}
2. Logistic Regresion
\begin{align}
\mathop{minimize\ }\limits_{w}{\sum_{i = 1}^{n}log(1 + exp(-y_i x_i^T w))}
\end{align}
3. Maximum likelihood estimation:
\begin{align}
\mathop{maximize\ }\limits_{\theta}{\sum_{i=1}^n{log p_\theta(x_i)}}
\end{align}  

# Convex
一般情况中，局部最小值并不是全局最小值，而且还有着各种各样的约束条件，因此先来研究一下凸优化问题。

## Convex Problems（凸问题）
凸优化问题有很好的性质,其局部最优解就是全局最优解，就是说，一旦你找到了一个局部最优解，
那么它一定是你能找到中的最好的（也就是全局最优的）。因此若可以把某一问题抽象为凸问题，那么会更容易求解。

### 一些概念
1. 凸集  
一个集合$$ C \subset R^n $$，任取两个元素$$ x,y \in C $$，并有一个实数$$ \alpha \in [0,1] $$，如果有
\begin{align}
\alpha x + (1-\alpha)y \in C
\end{align}
那么C为凸集(convex set)。可以看出，此集合中的任意两个点之间连线上的点也都在该集合中。

2. 凸函数  
如果一个函数满足以下两个条件，那么它是凸函数(convex function)：  
	1. 该函数定义域是凸集。（记作$$ D(f) $$）  
	2. 对任意两点$$ x,y \in D(f) $$，以及$$ \alpha \in [0,1] $$，有:  
	\begin{align}
	f(\theta x + (1-\theta)y)\le\theta f(x)+(1-\theta)f(y)
	\end{align}

	可以看出如果一个函数是一个凸函数，那么该函数两点的连线在该函数图形的上方。

3. 凸集与凸函数的关系  
定义一个函数f的epigraph为集合：
\begin{align}
epi f = \\{ (x,t):f(x)\le t \\}
\end{align}
那么有：
	1. epi f是凸集当且仅当f是凸函数
	2. 对于凸函数f，集合$$ \{ x:f(x) \le a \} $$是凸集

![](/images/optimization/1.png){: width="220px" height="210px"}  

凸函数举例如下:  
1. Linear functions:  
\begin{align}
f(x)=b^Tx+c
\end{align}
2. Quadratic functions：  
\begin{align}
f(x)={\frac 12}x^TAx+b^Tx+c，其中A为半正定或正定。
\end{align}
对于前面的regression，有：
\begin{align}
{\frac 12}{||Xw-y||}^2={\frac 12}w^TX^TXw-y^TXw+{\frac 12}y^Ty 
\end{align}
3. Norms($$ L_1 、L_2 $$)
\begin{align}
||\alpha x+(1-\alpha)y|| \le ||\alpha x||+||(1-\alpha)y||=\alpha ||x|| + (1-\alpha)||y||
\end{align}

## Convex Optimization Problems
一个优化问题是凸的仅当它的目标函数是凸函数，不等式约束$$ f_i $$是凸函数，等式约束$$ h_j $$是仿射函数
\begin{align}
\mathop{minimize\ }\limits_{x}f_0(x)  \ \ (凸函数)  \\\
s.t. \  f_i(x) &\le 0, i = \{1,...k\} \ \ (凸集)  \\\
h_j(x) &= 0, j = {1,...l} \ \ (仿射函数)
\end{align}

再说一下凸问题的优点：
1. 局部最小即全局最小
2. $$ \nabla f(x) = 0 $$当且仅当$$ x $$是$$ f(x) $$的全局极小值点

# Lagrange Duality(拉格朗日对偶)
我们先是把某一问题抽象成凸问题，接下来就是如何解决凸优化，拉格朗日对偶就是这样一个工具。
可以看到有了约束条件后求极值麻烦，而拉格朗日对偶可以简化解决问题的方法，把原始的约束问题通过拉格朗日函数转化为无约束问题。

对原始优化问题描述如下：  
\begin{align}
\mathop{minimize\ }\limits_{x}f_0(x) \\\
s.t. \  f_i(x) &\le 0, i = \{1,...k\} \\\
h_j(x) &= 0, j = {1,...l}
\end{align}

引入拉格朗日乘子(Lagrange multipliers)，$$ \vec \lambda = (\lambda_1,\lambda_2,...,\lambda_k)^T，\vec \nu = (\nu_1,\nu_2,...,\nu_l)^T ，并且\lambda_i \ge 0，\nu_i \in R $$ ：  
\begin{align}
L(x,\lambda,\nu) = f_0(x) + \sum_{i=1}^k{\lambda_i f_i(x)} + \sum_{j=1}^l{\nu_j h_j(x)}
\end{align}
其中，由不等式约束引入的KKT条件（$$ i=1,2,...,n $$）为：  
\begin{align}
\begin{cases}
f_i(x) \le 0; \\\
\lambda_i \ge 0; \\\
\lambda_i f_i(x) = 0.
\end{cases}
\end{align}

拉格朗日对偶函数g定义为（inf指下确界）：  
\begin{align}
g(\lambda,\nu) &= \mathop{inf \ }\limits_{x}L(x,\lambda,\nu) \\\
&= \mathop{inf\ }\limits_{x}{\left \\{ f_0(x) + \sum_{i=1}^k{\lambda_i f_i(x)} + \sum_{j=1}^l{\nu_j h_j(x)} \right \\} }
\end{align}

原始问题等价于（sup指上确界）：  
\begin{align}
\mathop{minimize \ }\limits_x { \left [ \mathop{sup}\limits_{\lambda \succeq 0, \nu }{L(x,\lambda,\nu)} \right ] }
\end{align}

为什么等价：  

1. 若某个$$ x $$违反了原始的约束，即有$$ f_i(x) \gt 0 或 h_j(x) \ne 0 $$，那么：  
\begin{align}
\mathop{max}\limits_{\lambda \ge 0,\nu}{\left [ f_0(x) + \sum_{i=1}^k{\lambda_i f_i(x)} + \sum_{j=1}^l{\nu_j h_j(x)} \right ]} = +\infty
\end{align}
因为若$$ f_i(x) \gt 0 $$，那么可以令$$ \lambda_i \to +\infty $$，从而使得$$ \lambda_i f_i(x) \to +\infty $$；若$$ h_j(x) \ne 0 $$，也可以让$$ \nu_j h_j(x) \to +\infty $$  

2. 若x满足原始的约束，那么：  
\begin{align}
\mathop{max}\limits_{\lambda \ge 0,\nu}{\left [ f_0(x) + \sum_{i=1}^k{\lambda_i f_i(x)} + \sum_{j=1}^l{\nu_j h_j(x)} \right ]} = \mathop{max}\limits_{\lambda \ge 0,\nu}{\left [  f_0(x) \right ]} = f_0(x)
\end{align}
因为x是个常量，变量是$$ \lambda 、\nu $$  

对偶问题则交换了min与max：
\begin{align}
\mathop{maximize \ }\limits_{\lambda \succeq 0, \nu}{\left [ \mathop{inf}\limits_{x}{L(x,\lambda,\nu)} \right ]}
\end{align}

介绍一下Duality gap：  

1. Duality gap 是原问题最优解(optimal primal value)与对偶问题最优解(optimal dual value)的差。
设原问题最优解为$$ p^{*} $$，对偶问题最优解为$$ d^{*} $$，那么duality gap为$$ p^{*} - d^{*} $$。这个值始终是$$ \ge 0$$的。  

2. duality gap为0当且仅当strong duality成立，否则则是weak duality（duality gap为正）.  
strong duality成立的情况下，我们可以通过求解dual problem来优化primal problem.  
一个优化问题，通过求出它的dual problem ，在只有weak duality成立的情况下，我们至少可以得到原始问题的一个下界。
而如果strong duality成立，则可以直接求解dual problem来解决原始问题。  

3. weak duality对于所有的优化问题都成立。  
无论primal problem是什么形式，dual problem总是一个convex optimization problem——它的极值是唯一的（如果存在的话）。  
这里简单证明一下weak duality, 即若有$$ \lambda \succeq 0 $$,那么：  
\begin{align}
g( \lambda, \nu ) \le f_0( x^{\*} ) ，x^{\*} 为f_0(x)的极小值点
\end{align}
因为
\begin{align}
g(\lambda, \nu) &= \mathop{inf \ }\limits_{x}{L(x,\lambda,\nu)} \\\ 
& \le L( x^{\*} ,\lambda,\nu ) \\\
&= f_0( x^{\*} ) + \sum_{i=1}^k{\lambda_i f_i( x^{\*} )} + \sum_{j=1}^l{\nu_j h_j( x^{\*} )} \\\
& \le f_0( x^{\*} )
\end{align}
可以看出对偶问题给出了原始问题最优值的下界。

介绍一下Slater条件:  

如果原始问题是凸问题，$$ f_i(x) $$是凸函数，$$ h_j(x) $$是仿射函数，
并且其可行域中至少有一个点使得不等式约束严格成立（是所有的不等式约束同时严格成立），此时强对偶成立。

介绍一下Complementary slackness（互补松弛性）：  

看看strong duality成立时候的一些性质。设$$ x^{*} $$是原问题的最优解，$$ ( \lambda^{*} , \nu^{*} ) $$是对偶问题的最优解，则有：  
\begin{align}
f_0( x^{\*} ) &= g( \lambda^{\*} , \nu^{\*} ) \\\
&= \mathop{inf \ }\limits_{x}{ \left \\{  f_0(x) + \sum_{i=1}^k{ \lambda_i^{\*} f_i(x)} + \sum_{j=1}^l{ \nu_j^{ \* } h_j(x)} \right \\} } \\\
& \le f_0( x^{\*} ) + \sum_{i=1}^m {\lambda_i^{\*} f_i(x^{\*})} + \sum_{j=1}^l{\nu_i^{\*}h_i(x^\*)} \\\
& = f_0(x^\*)
\end{align}
由上面可以得到：
\begin{align}
\sum_{i=0}^k{ \lambda_i^{\*} f_i( x^\* ) } = 0，其中i = 1,...,k
\end{align}
上述条件称为互补松弛性，它对任意原问题最优解$$ x^* $$都成立（当强对偶性成立时）。我们可以将互补松弛条件写成：
\begin{align}
\lambda_i^\* \gt 0 & \Rightarrow f_i(x^\*) = 0 \\\
f_i(x^\*) \lt 0 & \Rightarrow \lambda_i^\* = 0
\end{align}

介绍一下Karush-Kuhn-Tucker(简称KKT)条件：  

现在假设函数f0,⋯,fk,h1,⋯,hl可微，但是并不假设这些函数是凸函数。  

1. 非凸问题的KKT条件  
令$$ x^* $$是原问题的最优解，$$ (\lambda^*,\nu^*) $$是对偶问题的最优解，对偶间隙为零。
因为$$ L(x,\lambda^*,\nu^*) $$在$$ x^* $$处取得最小值，则有：
\begin{align}
\nabla f_0(x^\*) + \sum_{i=1}^k{\lambda_i^\* \nabla f_i(x^\*)} + \sum_{j=1}^l{\nu_i^\* \nabla h_j(x^\*)} = 0
\end{align}
因此我们有：  
\begin{align}
f_i(x^\*)  \le 0，i= 1,...,k \\\
h_j(x^\*)  = 0, j=1,...,l \\\
\lambda^\*  \ge 0, i=1,...,k \\\
\lambda_i^\* f_i(x^\*) = 0,i=1,...,k \\\
\nabla f_0(x^\*) + \sum_{i=1}^k{\lambda_i^\* \nabla f_i(x^\*)} + \sum_{j=1}^l{\nu_i^\* \nabla h_j(x^\*)} = 0
\end{align}
称上式为Karush-Kuhn-Tucker（KKT）条件。  
总之，对于目标函数和约束函数可微的任意优化问题，如果强对偶性成立，那么任何一对原问题最优解和对偶问题最优解必须满足KKT条件。  
即为强对偶性的必要条件。KKT条件是一组解成为最优解的必要条件。  

2. 凸问题的KKT条件  
当原问题是凸问题时，满足KKT条件的点也是原、对偶最优解。也就是说，如果函数$$ f_i $$是凸函数，$$ h_j $$是仿射函数，
$$ \tilde x, \tilde \lambda , \tilde \nu $$是任意满足 KKT 条件的点，也即：
\begin{align}
f_i(\tilde x)  \le 0，i= 1,...,k \\\
h_j(\tilde x)  = 0, j=1,...,l \\\
\tilde \lambda  \ge 0, i=1,...,k \\\
\tilde \lambda_i f_i(\tilde x) = 0,i=1,...,k \\\
\nabla f_0(\tilde x) + \sum_{i=1}^k{\tilde \lambda_i \nabla f_i(\tilde x)} + \sum_{j=1}^l{\tilde \nu_i \nabla h_j(\tilde x)} = 0
\end{align}
总之，对目标函数和约束函数可微的任意凸优化问题，任意满足KKT条件的点分别是原、对偶问题最优解，对偶间隙为零。  
即为强对偶性的充要条件。如果一个凸优化问题有可微的目标函数和约束，并且满足Slater条件，则KKT条件是取得最优的充要条件：
Slater条件保证了最优对偶间隙为零并且最优点可以取到。原始问题是凸的，则KKT条件也是充分的，这是因为KKT的最后一个条件
在对拉格朗日函数取下确界的时候成为了充要条件。

在满足了KKT条件后，有：
\begin{align}
\mathop{sup\ }\limits_{\lambda \succeq 0, \nu}{g(\lambda, \nu)} = f_0(x^\*)
\end{align}

### Intuition of Duality

可以把对偶解释为线性近似(linear approximation)

设  
\begin{align}
\mathbb I_{-}(a) = 
\begin{cases}
\infty ，a \gt 0 \\\
0，otherwise
\end{cases}
\end{align}

\begin{align}
\mathbb I_{0}(a) = 
\begin{cases}
0，a = 0 \\\
\infty ，otherwise
\end{cases}
\end{align}

把问题重新写为：  
\begin{align}
\mathop{minimize\ }\limits_{x}{f_0(x) + \sum_{i=1}^k{\mathbb I_{-}(f_i(x))} + \sum_{j=1}^l{\mathbb I_{0}(h_j(x))}}
\end{align}
接着用$$ \lambda_i f_i(x) $$ 替换 $$ \mathbb I_{-}(f_i(x)) $$，用$$ \nu_i h_j(x) $$ 替换 $$ \mathbb I_{0}(h_j(x)) $$

### Interpretation of the Lagrange multipliers

#### 等式约束
当目标函数加上等式约束条件之后，问题就变成如下形式：  
\begin{align}
\mathop{min \ }\limits_{x}{f_0(x)} \\\
s.t  \  h_j(x) &= 0，j=1,...,l
\end{align}
约束条件会将解的范围限定在一个可行域，此时不一定能找到使得$$ \nabla_x f(x) $$为0的点，
只需找到在可行域内使得$$ f(x) $$最小的值即可，常用的方法即为拉格朗日乘子法，引入$$ \nu \in R^m $$，得到：  
\begin{align}
L(x,\alpha) = f_0(x) + \sum_{j=1}^l{\nu_j h_j(x)}
\end{align}
看下图（这里$$ f(x,y) $$是目标函数，$$ g(x,y)=c $$是等式约束。在平面中画出$$ f(x,y) $$）

![](/images/optimization/2.png){: width="350px" height="250px"}  

目标函数$$ f(x,y) $$与约束$$ g(x,y) $$只有三种情况，相交、相切或者没有交集，没交集肯定不是解，只有相交或者相切可能是解，
但相交得到的一定不是最优值，因为相交意味着肯定还存在其它的等高线在该条等高线的内部或者外部，
使得新的等高线与目标函数的交点的值更大或者更小，这就意味着只有等高线与目标函数的曲线相切的时候，才可能得到可行解。

因此有：拉格朗日乘子法取得极值的必要条件是目标函数与约束函数相切，这时两者的法向量是平行的，即  
\begin{align}
\nabla_x f(x) - \nu \nabla_x h(x) = 0
\end{align}
只要满足上式，且满足之前的约束$$ h_j(x) = 0，j=1,...,l $$，即可得到解。

#### 不等式约束
考虑只有一个不等式约束的简单情况，给定如下不等式约束问题：  
\begin{align}
\mathop{min \ }\limits_{x}{f_0(x)} \\\
s.t  \  f_1(x) & \le 0
\end{align}
对应的拉格朗日函数：  
\begin{align}
L(x,\lambda) = f_0(x) + \lambda f_1(x)
\end{align}
如下图（$$ g(x) \le 0 $$为不等式约束, $$ f(x) $$为目标函数）

![](/images/optimization/3.png){: width="350px" height="250px"}  

由图可见可行解$$ x $$只能在$$ g(x)<0 $$或者$$ g(x)=0 $$的区域里取得：  
* 当可行解$$ x $$落在$$ g(x)<0 $$的区域内，此时直接极小化$$ f(x) $$即可
* 当可行解$$ x $$落在$$ g(x)=0 $$即边界上，此时等价于等式约束优化问题

当约束区域包含目标函数原有的的可行解时，此时加上约束可行解扔落在约束区域内部，
对应$$ g(x)<0 $$的情况，这时约束条件不起作用；当约束区域不包含目标函数原有的可行解时，
此时加上约束后可行解落在边界$$ g(x)=0 $$上。

下图分别描述了两种情况，右图表示加上约束可行解会落在约束区域的边界上（图中的小红点是任意标的）。

![](/images/optimization/4.png){: width="650px" height="250px"} 

要么可行解落在约束边界上即得$$ g(x)=0 $$，要么可行解落在约束区域内部，此时约束不起作用，
令$$ \lambda=0 $$消去约束即可，所以无论哪种情况都会得到：  
\begin{align}
\lambda g(x) = 0，（对应这里是\lambda f_1(x) = 0）
\end{align}

这里要说一下$$ \lambda $$的取值，在等式约束优化中，约束函数与目标函数的梯度只要满足平行即可，
而在不等式约束中则不然，若$$ \lambda \ne 0 $$，这便说明 可行解$$ x $$是落在约束区域的边界上的，
这时可行解应尽量靠近无约束时的解，所以在约束边界上，目标函数的负梯度方向应该远离约束区域朝向无约束时的解，
此时正好可得约束函数的梯度方向与目标函数的负梯度方向应相同：  
\begin{align}
-\nabla_x f(x) = \lambda \nabla_x g(x)，并且 \lambda \gt 0
\end{align}

如下图所示：  

![](/images/optimization/5.png){: width="650px" height="250px"}

可见对于不等式约束，只要满足一定的条件，依然可以使用拉格朗日乘子法解决，这里的条件便是KKT条件。

### Example

#### 例一  
最大化目标函数$$ u = x^3y^2z $$，约束为$$ x+y+z=12 $$

设$$ F(x,y,z) = x^3y^2z +\lambda(x+y+z-12) $$，  
\begin{align}  
\begin{cases}
F_x' = 3x^2y^2z + \lambda &= 0 \\\
F_y' = 2x^3yz + \lambda &= 0 \\\
F_z' = x^3y^2 + \lambda &= 0 \\\
x + y + z &= 0 
\end{cases}
\end{align}
解得唯一驻点(6,4,2)

#### 例二
最小化目标函数$$ x_1^2 -2x_1 + 1 + x_2^2 + 4x_2 + 4 $$，约束为$$ x_1 + 10x_2  > 10 $$和$$ 10x_1 -10x_2 < 10 $$  
先把约束变为：  
\begin{align}
-x_1 -10x_2 + 10 & \lt 0 \\\
10x_1 - 10x_2 - 10 & \lt 0
\end{align}
得到拉格朗日函数：  
\begin{align}
L(x,\alpha) &= f(x) + \alpha_1 g_1(x) + \alpha_2 g_2(x) \\\
&= x_1^2 -2x_1 + 1 + x_2^2 + 4x_2 + 4 + \alpha_1(-x_1 -10x_2 + 10) + \alpha_2(10x_1 - 10x_2 - 10)
\end{align}
则有：  
\begin{align}  
\begin{cases}  
-x_1 - 10x_2 + 10 & \lt 0 \\\
10x_1 - 10x_2 - 10 & \lt 0 \\\
\alpha_1 g_1(x) &= 0 \\\
\alpha_2 g_2(x) &= 0 \\\
L_{x_1}' = 2x_1 - 2 - \alpha_1 + 10 \alpha_2 &= 0 \\\
L_{x_2}' = 2x_2 + 4 - 10 \alpha_1 - 10 \alpha_2 &= 0
\end{cases}
\end{align}

对于$$ \alpha g(x) = 0 $$，要么$$ \alpha = 0 $$，要么$$ g(x) = 0 $$，分四种情况讨论：  

1. 若$$ \alpha_1 = \alpha_2 = 0 $$，则得到$$ x_1 = 1，x_2 = -1 $$，不满足第一个不等式，舍弃

2. 若$$ g_1(x) = g_2(x) = 0 $$，则得到$$ x_1 = 110/101，x_2 = 90/101 $$，进而解得$$ \alpha_1 = 58/101，\alpha_2 = 4/101 $$，满足条件

3. 其他两种情况解得不满足约束，舍弃。

## Optimization Algorithms

先介绍几个概念：  

1. 梯度  
\begin{align}
grad \ f(x_0,x_1,...,x_n) = (\frac{\partial f}{\partial x_0}, \frac{\partial f}{\partial x_1},...,\frac{\partial f}{\partial x_n})
\end{align}

2. Hessian matrix（海森矩阵）  
一个二阶可微函数$$ f:R^n \to R $$在一个点$$ x \in dom\ f $$是函数在该点的二阶导数：  
\begin{align}
H_{ij} = \frac {\partial^2 f}{\partial_{x_i}\partial_{x_j}}，1 \le i,j \le n.
\end{align}
函数f在点x处的海森矩阵可以表示为$$ \nabla^2 f(x) $$。由于$$ H_{ij} = H_{ji} $$，因此海森矩阵是对称矩阵。

3. 这里再说一下convex的定义：  
一个函数$$ f:R^n \to R $$是convex的，如果在每个点$$ x \in R^n $$，存在向量$$ \nabla f(x) \in R^n $$（被称作f的梯度(gradient)），使得
下列不等式对所有$$ y \in R^n $$成立：  
\begin{align}
f(y) \ge f(x) + \nabla f(x)^T(y-x)
\end{align}
	其中$$ \nabla f(x)^T(y-x) $$表示$$ \nabla f(x) $$与$$ y-x $$的点积(dot product)。
	线性函数$$ l_x(y) = f(x) + \nabla f(x)^T(y-x) $$可以被解释为f在点$$ (x,f(x)) $$处的切线。

### Gradient Descent
目标:  
\begin{align}
\mathop{minimize\ }\limits_{x}{f(x)}
\end{align}
只需要迭代：  
\begin{align}
x_{t+1}=x_t - \eta_t \nabla f(x_t)，其中\eta_t是步长(stepsize)
\end{align}
如何选择stepsize:  

1. Exact line search  
\begin{align}
\eta_t = \mathop{argmin\ }\limits_{\eta}{f(x-\eta \nabla f(x))}
\end{align}
通常不可行，计算量大

2. Fixed stepsize  
步长过大可能无法收敛

3. Backtracking line search  
设$$ \alpha \in (0,\frac 12)，\beta \in (0,1) $$  
在每次迭代时，不断的让$$ \eta = \beta \eta $$直到：  
\begin{align}
f(x - \eta \nabla f(x)) \le f(x) - \alpha \eta ||\nabla f(x)||^2
\end{align}

	如下图，可以看出当前点斜率越小，步长就越小：

	![](/images/optimization/7.png){: width="350px" height="250px"}

何时停止迭代：
通常$$ ||\nabla f(x)|| $$很小的时候，就可以停下来。
	
## Newton's method

牛顿法是为了求解函数值为零的时候变量的取值问题的，具体地，
当要求解$$ f(\theta)=0 $$时，如果f可导，那么可以通过迭代公式:  
\begin{align}
\theta := \theta - \frac {f(\theta)}{f'(\theta)}
\end{align}
来迭代求得最小值。通过一组图来说明这个过程:

![](/images/optimization/8.png){: width="650px" height="200px"} 

当应用于求解目标函数极值则变成$$ l'(\theta)=0 $$的问题。
这个与梯度下降不同，梯度下降的目的是直接求解目标函数极小值，
而牛顿法则通过求解目标函数一阶导为零的参数值，进而求得目标函数最小值。那么迭代公式写作：  
\begin{align}
\theta := \theta - \frac {l'(\theta)}{l''(\theta)}
\end{align}
	
当θ是向量时，牛顿法可以使用下面式子表示：  
\begin{align}
\theta := \theta - H^{-1}\nabla_{\theta}l(\theta)，其中H为海森矩阵
\end{align}

## Subgradient method

在优化问题中，我们可以对目标函数为凸函数的优化问题采用梯度下降法求解，但是在实际情况中，
目标函数并不一定光滑、或者处处可微，这时就需要用到次梯度（Subgradient）下降算法。  
凸函数是指如果函数f可微，那么当且仅当$$ dom(f) $$为凸集，
且对于$$ \forall x,y \in dom(f) $$，使得$$ f(y) \geq f(x)+\nabla f(x)^T(y−x) $$，
则函数f为凸函数。这里所说的次梯度是指在函数f上的点x满足以下条件的$$ g \in \mathbb{R}^n $$：  
\begin{align}
f(y) \geq f(x) + g^T(y-x)
\end{align}
其中，函数f不一定要是凸函数，非凸函数也可以，即对于凸函数或者非凸函数而言，
满足上述条件的g均为函数在该点的次梯度。  
凸函数的次梯度一定存在，如果函数f在点x处可微，那么$$ g=\nabla f(x) $$，为函数在该点的梯度，
且唯一；如果不可微，则次梯度不一定唯一。但是对于非凸函数，次梯度则不一定存在，也不一定唯一。  
如下图，在满足$$ f_1(x)=f_2(x) $$的点处，次梯度为任意一条直线在向量$$ \nabla f_1(x) $$和$$ \nabla f_2(x) $$之间。

![](/images/optimization/9.jpg){: width="450px" height="150px"}

次梯度具有以下优化条件（subgradient optimality condition）：对于任意函数f（无论是凸还是非凸），函数在点x处取得最值等价于：  
\begin{align}
f(x^{\ast})=\min \limits_{x} f(x) \, \Leftrightarrow \, 0 \in \partial f(x^{\ast})
\end{align}
即，当且仅当0属于函数f在点$$ x^{\ast} $$处次梯度集合的元素时，$$ x^{\ast} $$为最优解。  
次梯度算法与梯度下降算法类似，仅仅用次梯度代替梯度，即：  
\begin{align}
x^{(k)} = x^{(k-1)} - t_k \cdot g^{(k-1)}, \, k=1,2,3,\ldots
\end{align}
其中，$$ g^{(k-1)} \in \partial f(x^{(k-1)}) $$，为f(x)在点x处的次梯度。
与梯度下降算法不同的地方在于，次梯度算法并不是下降算法，每次对于参数的更新
并不能保证代价函数是呈单调递减的趋势，因此，一般情况下我们选择：  
\begin{align}
f(x_{best}^{(k)}) = \min \limits_{i=0,\ldots,k} f(x^{(i)})
\end{align}
次梯度算法没有明确的步长选择方法，只有步长选择准则，具体如下。
次梯度算法并不像梯度下降一样，可以在每一次迭代过程中自适应的计算此次步长（adaptively computed），
而是事先设定好的（pre-specified）。  
（1） Fixed step sizes $$ t_k = t, \, all \, k = 1,2,3,\ldots $$  
（2） Diminishing step sizes： 选择满足以下条件的$$ t_k $$:
\begin{align}
\sum_{k=1}^{\infty} t_k^2 < \infty, \, \sum_{k=1}^{\infty} t_k = \infty
\end{align}

## 参考资料
[Quadratic Forms and Convexity](/docs/optimization/1.pdf)  
[Machine Learning and Optimization](http://freemind.pluskid.org/series/mlopt/)  
[Convex Optimization: Algorithms and Complexity](/docs/optimization/3.pdf)  
[Introduction to Convex Optimization for Machine Learning](/docs/optimization/2.pdf)  
[Lagrange duality](http://www.cnblogs.com/90zeng/p/Lagrange_duality.html)  
[Convex function](https://en.wikipedia.org/wiki/Convex_function)  
[Duality gap](https://en.wikipedia.org/wiki/Duality_gap)  
[Weak duality](https://en.wikipedia.org/wiki/Weak_duality)  
[Duality](http://blog.pluskid.org/?p=702)  
[仿射变换](https://www.zhihu.com/question/20666664)  
[拉格朗日对偶性](http://legend4917.github.io/2015/12/17/%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E5%AF%B9%E5%81%B6%E6%80%A7/)  
[约束优化方法之拉格朗日乘子法与KKT条件 ](http://www.cnblogs.com/ooon/p/5721119.html)  
[Unconstrained Minimization](http://www.csc.kth.se/utbildning/kth/kurser/DD3364/Lectures/KKT.pdf)  
[Gradient Desent Revisited](https://www.cs.cmu.edu/~ggordon/10725-F12/scribes/10725_Lecture5.pdf)  
[strongly convex](http://www.cs.cornell.edu/courses/cs4820/2014sp/notes/dscnt.pdf)  
[Convexity and Optimization](http://www.stat.cmu.edu/~larry/=sml/convexopt.pdf)  
[Subgradient](http://www.hanlongfei.com/convex/2015/10/02/cmu-10725-subgradidient/)  
[次梯度法](http://www.wikiwand.com/zh-hk/%E6%AC%A1%E6%A2%AF%E5%BA%A6%E6%B3%95)

