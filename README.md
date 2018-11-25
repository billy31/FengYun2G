# FengYun2G
## 2018.11.11
Try Git to control versions
## 2018.11.12
Generate stable mir values through tir sequences data
## 2018.11.20
**Basic flowchar**
```mermaid
graph TD;
st=>input tiled data
op1=>generate stable values:method(?)
cond1=>condition:Yes or No?
op2=>calculate dynamic threshold:method(?)
cond2=>condition:Yes or No?
op3=>calculate contextual values:method(winsize=5)
cond3=>condtion:Yes or No?
cond4=>values:No fire
e=>end
st->op1->cond1->op2->cond2->op3->cond3
cond1(yes)->op2
cond2(yes)->op3
cond3(yes)->e
cond1(no)->cond4
cond2(no)->cond4
cond3(no)->cond4
```
## 2018.11.25
1.Generate stable mir values through tir data within [270, 330]
2.Considering revisiing judgement for stable values (errors in current results and unavailable of Kalman filter method)

