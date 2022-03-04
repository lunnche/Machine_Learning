# The Fundamentals of Machine Learning

the main regions and the most notable landmarks of ML:
* supervised or  unsupervised
* online or batch learning
* instance-based or model-based  

definition of ML:

Machine Learning is the science(and art) of programming computers so they can learn from data.  

Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed.  --Arthur Samuel,1959  

A comuter program is said to learn from expericence E with respect to some task T and some performance measure P,if its performance on T,as measured by P,improves with experience E.  --Tom Mitchell,1997

第三种定义翻一下：
一个计算机程序利用经验E来学习任务T，性能是P，如果针对任务T的性能P随着经验E不断增长，则称为机器学习。  

为啥要用机器学习？
先看看，不用机器学习的话，怎么写识别垃圾邮件的程序？
1. 找垃圾邮件的特点
2. 为垃圾邮件的每一个特点写检测算法
3. 测试程序并重复做上述两部，知道结果足够好。  

![image-20220303091319550](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220303091319550.png)

这种做法有个弊端：
如果问题非常复杂，你需要写的用来检测垃圾邮件的规则会非常多，这很难维护。  

相反，用机器学习来做垃圾邮件分类，通过比较普通邮件（ham）和垃圾邮件，自动学习出哪些特点标志着垃圾邮件，写出的程序会更短，更易于维护，更准确。

![image-20220303091837494](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220303091837494.png)

如果垃圾邮件发送者，针对你的传统算法进行规避，比如把你能够检测到的关键词“4u”改为“for u”，那么你就要一直更新你的算法。
而机器学习算法会自动注意到“for u”在用户手动标记的垃圾邮件中频繁出现，无须人工干预即可自动标记出包含“for u”的垃圾邮件。  

机器学习另一个亮点是擅长处理对于传统方法而言太复杂或没有已知算法的问题。例如，对于语音识别，假设你想写一个可以识别 “one”和“two”的简单程序。你可能注意到“two”的起始是一个高 音（“T”），因此会写一个可以测量高音强度的硬编码算法，用于区 分“one”和“two”。但是很明显，这个方法不能推广到所有的语音识 别（人们所处环境不同、语言不同、使用的词汇不同）。（现在）最佳 的方法是根据给定的大量单词录音，写一个可以自我学习的算法。

例如，对于语音识别，假设你想写一个可以识别 “one”和“two”的简单程序。你可能注意到“two”的起始是一个高 音（“T”），因此会写一个可以测量高音强度的硬编码算法，用于区 分“one”和“two”。但是很明显，这个方法不能推广到所有的语音识 别（人们所处环境不同、语言不同、使用的词汇不同）。（现在）最佳 的方法是根据给定的大量单词录音，写一个可以自我学习的算法。

最后，机器学习可以帮助人类进行学习（见图1-4）。机器学习算 法可以检测自己学到了什么（尽管这对于某些算法很棘手）。例如，在 垃圾邮件过滤器训练了足够多的垃圾邮件后，就可以用它列出垃圾邮件 预测器的单词和单词组合。有时可能会发现不引人关注的关联或新趋 势，这有助于更好地理解问题。使用机器学习方法挖掘大量数据来帮助 发现不太明显的规律。这称作数据挖掘。

![image-20220303092913238](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220303092913238.png)

总结一下：
机器学习适用于：
·有解决方案（但解决方案需要进行大量人工微调或需要遵循大量 规则）的问题：机器学习算法通常可以简化代码，相比传统方法有更好 的性能。
·传统方法难以解决的复杂问题：最好的机器学习技术也许可以找 到解决方案。
·环境有波动：机器学习算法可以适应新数据。
·洞察复杂问题和大量数据。


## 机器学习能处理的问题：
一大堆，不一一列了，和预测有关的：
Forecasting your company’s revenue next year, based on many performance metrics. This a regression task (i.e., predicting values), which may be tackled using any regression model, such as a Linear Regression or Polynomial Regression model (see Chapter 4), a regression SVM (see Chapter 5), a regression random forest (see Chapter 7) or an artificial neural network (see Chapter 10). If you want to take into account sequences of past performance metrics, you may want to use recurrent neural networks (RNNs), convolutional neural networks (CNNs) or Transformers (see Chapter 15 and Chapter 16).

还有这下面这个我也觉得是很泛用很酷的东西：
Representing a complex, high-dimensional dataset in a clear and insightful diagram: this is data visualization, often involving dimensionality reduction techniques (see Chapter 8).

来了来了，阿尔法狗：
Building an intelligent bot for a game. This is often tackled using Reinforcement Learning (RL, see Chapter 18), which is a branch of Machine Learning that trains agents (such as bots) to pick the actions that will maximize their rewards over time (e.g., a bot may get a reward every time the player loses some life points), within a given environment (e.g., the game). The famous AlphaGo program that beat the world champion at the game of go was built using RL.

## Types of Machine Learning Systems
·是否在人类监督下训练（有监督学习、无监督学习、半监督学习 和强化学习）。

·是否可以动态地进行增量学习（在线学习和批量学习）。

·是简单地将新的数据点和已知的数据点进行匹配，还是像科学家 那样，对训练数据进行模式检测然后建立一个预测模型（基于实例的学 习和基于模型的学习）。

For example, a state-of-the-art spam filter may learn on the fly using a deep neural network model trained using examples of spam and ham; this makes it an online, model-based, supervised learning system.

## Supervised/Unsupervised Learning
根据训练期间接受的监督数量和监督类型，可以将机器学习系统分 为以下四个主要类别：有监督学习、无监督学习、半监督学习和强化学习。

### Supervised learning
In supervised learning, the training set you feed to the algorithm includes the desired solutions, called **labels** (Figure 1-5).

attribute 和 feature的区别：
In Machine Learning an attribute is a data type (e.g., “Mileage”), while a feature has several meanings, depending on the context, but generally means an attribute plus its value (e.g., “Mileage = 15,000”). Many people use the words attribute and feature interchangeably.

一些监督学习算法：
k-Nearest
Neighbors 
Linear Regression 
Logistic Regression 
Support Vector Machines (SVMs) 
Decision Trees and Random Forests 
Neural networks

### Unsupervised learning
In unsupervised learning, as you might guess, the training data is unlabeled (Figure 1-7). The system tries to learn without a teacher.
一些无监督学习技术：
Clustering 聚类
K-Means k-均值
DBSCAN 
Hierarchical Cluster Analysis (HCA) 分层聚类分析 
Anomaly detection and novelty detection 异常检测和新颖性检测
One-class SVM 单类SVM
Isolation Forest 孤立森林
Visualization and dimensionality reduction 可视化和降维 
Principal Component Analysis (PCA) 主成分分析
Kernel PCA 核主成分分析
Locally-Linear Embedding (LLE) 局部线性嵌入 
t-distributed Stochastic Neighbor Embedding (t-SNE) t-分布随机近邻嵌入 
Association rule learning 关联规则学习
Apriori 
Eclat

可视化算法也是无监督学习算法的一个不错的示例：你提供大量复 杂的、未标记的数据，算法轻松绘制输出2D或3D的数据表示（见图19）。这些算法会尽其所能地保留尽量多的结构（例如，尝试保持输入 的单独集群在可视化中不会被重叠），以便于你理解这些数据是怎么组 织的，甚至识别出一些未知的模式。

![image-20220303105345916](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220303105345916.png)

与之相关的一个任务是降维，降维的目的是在不丢失太多信息的前 提下简化数据。方法之一是将多个相关特征合并为一个。例如，汽车里 程与其使用年限存在很大的相关性，所以降维算法会将它们合并成一个 代表汽车磨损的特征。这个过程叫作特征提取。**feature extraction**

## Semisupervised learning
Since labeling data is usually time-consuming and costly, you will often have plenty of unlabeled instances, and few labeled instances. Some algorithms can deal with data that’s partially labeled. This is called semisupervised learning (Figure 1-11).

![image-20220303112643750](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220303112643750.png)

大多数半监督学习算法是无监督算法和有监督算法的结合。例如， 深度信念网络（DBN）基于一种互相堆叠的无监督组件，这个组件叫作 受限玻尔兹曼机（RBM）。受限玻尔兹曼机以无监督方式进行训练，然 后使用有监督学习技术对整个系统进行微调。

## Reinforcement Learning
Reinforcement Learning is a very different beast. The learning system, called an agent in this context, can observe the environment, select and perform actions, and get rewards in return (or penalties in the form of negative rewards, as in Figure 1-12). It must then learn by itself what is the best strategy, called a policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.

![image-20220303141311306](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220303141311306.png)


DeepMind的 AlphaGo项目也是一个强化学习的好示例。2017年5月，AlphaGo在围棋 比赛中击败世界冠军柯洁而声名鹊起。通过分析数百万场比赛，然后自 己跟自己下棋，它学到了制胜策略。要注意，在跟世界冠军对弈的时 候，AlphaGo处于关闭学习状态，它只是应用它所学到的策略而已。

## Batch and Online Learning
另一个给机器学习系统分类的标准是看系统是否可以从传入的数据 流中进行增量学习。

## Batch learning
In batch learning, the system is incapable of learning incrementally: it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called offline learning.

If you want a batch learning system to know about new data (such as a new type of spam), you need to train a new version of the system from scratch on the full dataset (not just the new data, but also the old data), then stop the old system and replace it with the new one.

Fortunately, the whole process of training, evaluating, and launching a Machine Learning system can be automated fairly easily (as shown in Figure 1-3), so even a batch learning system can adapt to change. Simply update the data and train a new version of the system from scratch as often as needed.

This solution is simple and often works fine, but training using the full set of data can take many hours, so you would typically train a new system only every 24 hours or even just weekly. If your system needs to adapt to rapidly changing data (e.g., to predict stock prices), then you need a more reactive solution.

Also, training on the full set of data requires a lot of computing resources (CPU, memory space, disk space, disk I/O, network I/O, etc.). If you have a lot of data and you automate your system to train from scratch every day, it will end up costing you a lot of money. If the amount of data is huge, it may even be impossible to use a batch learning algorithm.

Finally, if your system needs to be able to learn autonomously and it has limited resources (e.g., a smartphone application or a rover on Mars), then carrying around large amounts of training data and taking up a lot of resources to train for hours every day is a showstopper.

Fortunately, a better option in all these cases is to use algorithms that are capable of learning incrementally.

## Online learning
In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or by small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives (see Figure 1-13).

![image-20220303143156585](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220303143156585.png)

Online learning is great for systems that receive data as a continuous flow (e.g., stock prices) and need to adapt to change rapidly or autonomously. It is also a good option if you have limited computing resources: once an online learning system has learned about new data instances, it does not need them anymore, so you can discard them (unless you want to be able to roll back to a previous state and “replay” the data). This can save a huge amount of space.

Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machine’s main memory (this is called out-of-core learning). The algorithm loads part of the data, runs a training step on that data, and repeats the process until it has run on all of the data (see Figure 1-14).

Out-of-core learning is usually done offline (i.e., not on the live system), so online learning can be a confusing name. Think of it as incremental learning.

![image-20220303144007475](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220303144007475.png)

One important parameter of online learning systems is how fast they should adapt to changing data: this is called the learning rate. If you set a high learning rate, then your system will rapidly adapt to new data, but it will also tend to quickly forget the old data (you don’t want a spam filter to flag only the latest kinds of spam it was shown). Conversely, if you set a low learning rate, the system will have more inertia; that is, it will learn more slowly, but it will also be less sensitive to noise in the new data or to sequences of nonrepresentative data points (outliers).

A big challenge with online learning is that if bad data is fed to the system, the system’s performance will gradually decline. If we are talking about a live system, your clients will notice. For example, bad data could come from a malfunctioning sensor on a robot, or from someone spamming a search engine to try to rank high in search results. To reduce this risk, you need to monitor your system closely and promptly switch learning off (and possibly revert to a previously working state) if you detect a drop in performance. You may also want to monitor the input data and react to abnormal data (e.g., using an anomaly detection algorithm).

## Instance-Based Versus Model-Based Learning
One more way to categorize Machine Learning systems is by how they generalize（泛化）  

Most Machine Learning tasks are about making predictions. This means that given a number of training examples, the system needs to be able to generalize to examples it has never seen before. Having a good performance measure on the training data is good, but insufficient; the true goal is to perform well on new instances.

## Instance-based learning
instance-based learning: the system learns the examples by heart, then generalizes to new cases by using a similarity measure to compare them to the learned examples (or a subset of them). For example, in Figure 1-15 the new instance would be classified as a triangle because the majority of the most similar instances belong to that class.

![image-20220303150437233](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220303150437233.png)

## Model-based learning
Another way to generalize from a set of examples is to build a model of these examples and then use that model to make predictions. This is called model-based learning (Figure 1-16).

来个栗子
使用Scikit-Learn训练并运行一个线性模型
```python
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import sklearn.linear_model

# Load the data 
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',') 
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t',encoding='latin1', na_values="n/a")

# Prepare the data 
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita) X = np.c_[country_stats["GDP per capita"]] y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data

country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')

plt.show()

# Select a linear model 
model = sklearn.linear_model.LinearRegression()

# Train the model model.fit(X, y)

# Make a prediction for Cyprus 
X_new = [[22587]] # Cyprus' GDP per capita 
print(model.predict(X_new)) # outputs [[ 5.96242338]]
```

## Main Challenges of Machine Learning  
## Insufficient Quentity of Training Data
对于复杂问题而言，数据比算法重要。

但：中小型数据集依然非常普遍，获得 额外的训练数据并不总是一件轻而易举或物美价廉的事情，所以暂时先 不要抛弃算法。

## Nonrepresentative Training Data
为了很好地实现泛化，至关重要的一点是对于将要泛化的新示例来 说，训练数据一定要非常有代表性。无论你使用的是基于实例的学习还 是基于模型的学习，都是如此。

It is crucial to use a training set that is representative of the cases you want to generalize to. This is often harder than it sounds: if the sample is too small, you will have sampling noise (i.e., nonrepresentative data as a result of chance), but even very large samples can be nonrepresentative if the sampling method is flawed. This is called sampling bias.（采样偏差）。

## Poor-quality Data
## Irrelevant Feautures
## Overfitting the Training Data
Overfitting happens when the model is too complex relative to the amount and noisiness of the training data

Here are possible solutions:
Simplify the model by selecting one with fewer parameters (e.g., a linear model rather than a high-degree polynomial model), by reducing the number of attributes in the training data or by constraining the model.

Gather more training data.

Reduce the noise in the training data (e.g., fix data errors and remove outliers).

Constraining a model to make it simpler and reduce the risk of overfitting is called **regularization**.正则化

例如，我们前面定义的线性模型有两个参数：θ0和θ1。因此， 该算法在拟合训练数据时，调整模型的自由度就等于2，它可以调整线 的高度（θ0 ）和斜率（θ1 ）。如果我们强行让θ1 =0，那么算法的自 由度将会降为1，并且拟合数据将变得更为艰难——它能做的全部就只 是将线上移或下移来尽量接近训练实例，最后极有可能停留在平均值附 近。这确实太简单了！如果我们允许算法修改θ1 ，但是我们强制它只 能是很小的值，那么算法的自由度将位于1和2之间，这个模型将会比自 由度为2的模型稍微简单一些，同时又比自由度为1的模型略微复杂一 些。你需要在完美匹配数据和保持模型简单之间找到合适的平衡点，从 而确保模型能够较好地泛化。

一个具体栗子看看正则化作用：
图1-23显示了三个模型。点线表示的是在以圆圈表示的国家上训练 的原始模型（没有正方形表示的国家），虚线是我们在所有国家（圆圈 和方形）上训练的第二个模型，实线是用与第一个模型相同的数据训练 的模型，但是有一个正则化约束。可以看到，正则化强制了模型的斜率 较小：该模型与训练数据（圆圈）的拟合不如第一个模型，但它实际上 更好地泛化了它没有在训练时看到的新实例（方形）。

![image-20220304085811028](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220304085811028.png)

在学习时，应用正则化的程度可以通过一个超参数来控制。超参数 是学习算法（不是模型）的参数。因此，它不受算法本身的影响。超参 数必须在训练之前设置好，并且在训练期间保持不变。如果将正则化超 参数设置为非常大的值，会得到一个几乎平坦的模型（斜率接近零）。 学习算法虽然肯定不会过拟合训练数据，但是也更加不可能找到一个好 的解决方案。调整超参数是构建机器学习系统非常重要的组成部分

## Underfitting the Training Data
，欠拟合和过拟合正好相反。它的产生通常是因 为对于底层的数据结构来说，你的模型太过简单。例如，用线性模型来 描述生活满意度就属于欠拟合。现实情况远比模型复杂得多，所以即便 是对于用来训练的示例，该模型产生的预测都一定是不准确的。

解决这个问题的主要方式有：

·选择一个带有更多参数、更强大的模型。

·给学习算法提供更好的特征集（特征工程）。

·减少模型中的约束（例如，减少正则化超参数）。

## Stepping Back
let’s step back and look at the big picture:

Machine Learning is about making machines get better at some task by learning from data, instead of having to explicitly code rules.

There are many different types of ML systems: supervised or not, batch or online, instance-based or modelbased.

In a ML project you gather data in a training set, and you feed the training set to a learning algorithm. If the algorithm is model-based, it tunes some parameters to fit the model to the training set (i.e., to make good predictions on the training set itself), and then hopefully it will be able to make good predictions on new cases as well. If the algorithm is instance-based, it just learns the examples by heart and generalizes to new instances by using a similarity measure to compare them to the learned instances.

The system will not perform well if your training set is too small, or if the data is not representative, is noisy, or is polluted with irrelevant features (garbage in, garbage out). Lastly, your model needs to be neither too simple (in which case it will underfit) nor too complex (in which case it will overfit).

## Testing and Validating


