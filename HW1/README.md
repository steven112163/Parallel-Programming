# Parallel Programming Assignment 1



###### tags: `平行程式作業`
[TOC]



## Part 1
### Q1-1
|Width=2|Width=4|Width=8|Width=16|
|---|---|---|---|
|![](https://i.imgur.com/UcqHYjI.png)|![](https://i.imgur.com/9k9HqIL.png)|![](https://i.imgur.com/5LJ39Ez.png)|![](https://i.imgur.com/F3OxhDR.png)|

|Width|Vector Utilization|
|---|---|
|2|89.8%|
|4|87.8%|
|8|86.7%|
|16|86.2%|

vector utilization會隨著VECTOR_WIDTH上升而下降，主要原因是因為處理指數的關係(如下圖)，隨著迴圈增加，非0的count會愈來愈少，等同lane會下降，導致vector utilization下降。
![](https://i.imgur.com/ZiC8QEq.png)




## Part 2
### Q2-1
```cpp=
a = (float *)__builtin_assume_aligned(a, 32);
b = (float *)__builtin_assume_aligned(b, 32);
c = (float *)__builtin_assume_aligned(c, 32);
```
根據[intel](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_storeu_ps&expand=6874), vmovups是代表single，也就是一個vector是32 bits，所以把16改成32即可。



### Q2-2
```shell=
# case 1
$ make clean && make && ./test_auto_vectorize -t 1
8.25947s
```
```shell=
# case 2
$ make clean && make VECTORIZE=1 && ./test_auto_vectorize -t 1
2.62275s
```
```shell=
# case 3
$ make clean && make VECTORIZE=1 AVX2=1 && ./test_auto_vectorize -t 1
1.40321s
```
|non-vectorized -> vectorized|3x|
|---|---|
|**vectorized -> AVX2**|**2x**|

vmovups或vmovaps都是使用單精度，也就是32 bits，而使用vector後可以加速約3倍，同時vector通常會是2的指數倍，所以default vector registers應該是128 bits。
而AVX2比使用vector後快了約2倍，所以AVX2 vector registers應該是256 bits。



### Q2-3
compiler會在第一版先將a寫入c，接下來才會判斷b是否大於a以及是否將b寫入c，所以他會認為if不一定成立而將b寫入c，因此無法產生像第二版一樣的assembly。但在第二版他反而是先判斷a和b哪個較大，再將較大的寫入c，所以compiler在第二版可以將較大的element寫入vector再寫入c。