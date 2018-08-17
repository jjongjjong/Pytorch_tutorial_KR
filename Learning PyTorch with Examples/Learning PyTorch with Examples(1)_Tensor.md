## [Tensors](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#id13)

### [Warm-up: numpy](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#id14)

*Before introducing PyTorch, we will first implement the network using numpy.*

Pytorch를 소개하기 앞서 numpy 라이브러리를 활용한 네크워크 구성을 먼저 해보자. 

*Numpy provides an n-dimensional array object, and many functions for manipulating these arrays. Numpy is a generic framework for scientific computing; it does not know anything about computation graphs, or deep learning, or gradients. However we can easily use numpy to fit a two-layer network to random data by manually implementing the forward and backward passes through the network using numpy operations:*

Numpy는 N-차원의 배열 객체를 만들 수 있으며, 이런 배열들을 조작할 수 있는 다양한 기능들을 제공한다.  Numpy 는 과학적 컴퓨팅 연산을 위한 프레임웤이다. (앞으로 배우게 될 파이토치와는 다르게) 그래프 연산, 딥러닝, 그래디언트는 고려되지 않고 만들어진 정말 연산을 위한 라이브러리다. 하지만 Numpy만으로도 순전파, 역전파를 구현하고, 임의의 데이터를 학습하는 신경망(2-layers)을 최적화 시킬 수 있다. 

![2_layer](C:\Users\jjangjjong\Google 드라이브\종환\블로그\pytorch_tutorial_KR\Learning PyTorch with Examples\pictures\2_layer.png)

```
# -*- coding: utf-8 -*-
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)    #임의의 입력값
y = np.random.randn(N, D_out)   #임의의 타겟값

# Randomly initialize weights
w1 = np.random.randn(D_in, H)	#w1~N(0,1)
w2 = np.random.randn(H, D_out)	#w2~N(0,1)

learning_rate = 1e-6  #얼마나 업데이트 시킬 지  

for t in range(500):
    # Forward pass: compute predicted y
    # 단순신경만은 단순 matrix 곱으로 표현될 수 있다. 
    #자세히 알고 싶다면 아래 참고 링크를 읽어보시길 권한다. 
    
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    # 손실함수 얼마나 부족한지를 계산.
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    # 역전파 계산하기. 미분의 연쇄법칙을 이용한 기울기 계산. 
    #  loss = (y_pred-y)^2 = ((w2*h_relu)-y)^2=(w2*(r(h))-y)^2=(w2(r(w1*x))-y)^2
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred) # Loss를 줄이기 위해 w2가 업데이트 되어야 할 방향
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h) # Loss를 줄이기 위해 w1이 업데이트 되어야할 방향

    # Update weights
    w1 -= learning_rate * grad_w1  # 방향에 어느 정도를 줄 지를 곱해서 가중치 업데이트
    w2 -= learning_rate * grad_w2  # 엄밀히 말하면 -grad가 업데이트 되어야할 방향이다.
```

[참고1 단순신경망](http://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220948258166&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView)

[참고2 역전파](http://jaejunyoo.blogspot.com/2017/01/backpropagation.html)

### [PyTorch: Tensors](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#id15)

*Numpy is a great framework, but it cannot utilize GPUs to accelerate its numerical computations. For modern deep neural networks, GPUs often provide speedups of [50x or greater](https://github.com/jcjohnson/cnn-benchmarks), so unfortunately numpy won’t be enough for modern deep learning.*

Numpy는 좋은 프레임웤이지만 아주 빠른 연산을 위한 GPU활용이 불가능하다. 최근의 딥러닝을 학습하기 위해선  아쉽게도 Numpy로는 충분치 않다ㅠ GPU는 약 50배 이상의 속도를 내게 해준다. 

*Here we introduce the most fundamental PyTorch concept: the **Tensor**. A PyTorch Tensor is conceptually identical to a numpy array: a Tensor is an n-dimensional array, and PyTorch provides many functions for operating on these Tensors. Behind the scenes, Tensors can keep track of a computational graph and gradients, but they’re also useful as a generic tool for scientific computing.*

여기선 Pytorch가장 기본적인 개념인 텐서(Tensor)에 대해 소개하고자 한다. 사실 텐서는 Numpy의 배열 array와 개념적으로 똑같다. 텐서는 N-차원의 배열이며, 이에 대해 다양한 텐서 연산을 지원한다. 아시다피시 텐서는 연산 그래프를 추적하거나, 기울기를 저장할 수 있지만, 그 외에도 과학적 연산과 같은 범용적 용도로도 활용될 수 있다. 

*Also unlike numpy, PyTorch Tensors can utilize GPUs to accelerate their numeric computations. To run a PyTorch Tensor on GPU, you simply need to cast it to a new datatype.*

Numpy와는 다르게 Pytorch의 텐서는 GPU를 활용할 수도 있다. GPU를 활용하려면 간단히 Datatype을 새롭게 정의해주면 된다. (.cuda 뒤에 배우게 될 것이다.)

*Here we use PyTorch Tensors to fit a two-layer network to random data. Like the numpy example above we need to manually implement the forward and backward passes through the network:*

여기선 PyTorch 텐서를 이용해서 위와 같은 2-layer 신경망을 학습시키고자 한다. 앞선 Numpy예제와 같이 순전파와 역전파를 구현하여야 한다. 

```
# -*- coding: utf-8 -*-
# 아래는 위 연산과 개념은 같으므로 주석은 생략하였다. 
import torch


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```