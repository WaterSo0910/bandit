---
title: 'Scalable bandit problem'
disqus: hackmd
---

# Scalable Bandit Problem
> 2021交大資工系專題研究

> 組員：蔡育呈、王耀德、周俊毅
>
> 指導老師：謝秉鈞
## Background
In the latest research, we could found a better choice(Neural Network) for applying on recommendation system. However, its cost is too large so that it haven't been popular nowadays.

## Problem description
Let's take Netflix recommendation system as an example,  we consider 3 main conditions which may cause tremendous cost:
1. The number of users (T)
2. The size of parameters in NN (Theta)
3. The number of movies (Arm)
## Algorithm implementation
![](https://i.imgur.com/2YStxKI.png)

## Contribution
We proposed 3 methods:
1. lazy update
2. ons
3. hash + inner product

## Result
- Total speed
![](https://i.imgur.com/9xlayWU.png)

- 3 extreme case vs original case
![](https://i.imgur.com/4AsVtlM.png)
