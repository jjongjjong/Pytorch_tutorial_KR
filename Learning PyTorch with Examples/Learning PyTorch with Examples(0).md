# Learning PyTorch with Examples(0)

**저자**: [Justin Johnson](https://github.com/jcjohnson/pytorch-examples)

**역자**: 장종환

This tutorial introduces the fundamental concepts of [PyTorch](https://github.com/pytorch/pytorch) through self-contained examples.

이 튜토리얼은 자체 코드를 통해 파이토치의 기본적인 개념을 소개하고자 한다. 

At its core, PyTorch provides two main features:

핵심적으로 파이토치는 2가지의 중요한 특징인 있는데 이는 아래와 같다. 

- An n-dimensional Tensor, similar to numpy but can run on GPUs

  Numpy와 같이 다차원 텐서를 지원하면서, 이에 더하여 GPU에서도 연산이 된다.

- Automatic differentiation for building and training neural networks

  인공신경망을 구축하고 학습시키기 위한 자동미분을 제공한다. 

We will use a fully-connected ReLU network as our running example. The network will have a single hidden layer, and will be trained with gradient descent to fit random data by minimizing the Euclidean distance between the network output and the true output.

우리 예제에서는 fully-connected ReLU network를 사용한다. 이 네트워크는 단일 신경망을 가지고, 경사하강법(gradient-descent)를 이용하여 학습된다.  데이터는 랜덤하세 생성하여 생성된 데이터와 네크워크의 아웃풋의 차이(유클리디안 거리)를 최소화하는 방향으로 이를 학습시키고자 한다. 

