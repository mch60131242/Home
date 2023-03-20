#!/usr/bin/env python
# coding: utf-8

# In[51]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


data_set = np.loadtxt('./data/ThoraricSurgery3.csv', delimiter=',')


# In[10]:


data_set.shape # 470개의 행과 17개의 특성 


# In[17]:


display(data_set[:10,0:16])


# In[20]:


X =data_set[:,0:16] #train
Y= data_set[:,16]# target


# In[21]:


print(X.shape)
print(Y.shape)


# In[26]:


## 층을 쌓는다

model = Sequential() ## 모델설정 

model.add(Dense(30,input_dim=16, activation='relu'))
#model.add(Dense(20, activation='relu'))## 층 추가, 노드(출력 층)가 30개, 칼럽의 갯수(16개), 활성함수 입력
model.add(Dense(1,activation='sigmoid'))## 다음층은 출력층이 1개   
 
model.summary()## 모델 요약 ## 510개의 파라미터인 이유 30개의 노드x 16개의 input = 480+ 가중치 30개
## 30x20 + 가중치 20개
## 세번째 층이 31개의 파리미터인 이유는 30개의 노드 + 1개의 가중치



# In[28]:


model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])
history =model.fit(X,Y, epochs=5, batch_size=16)  ## 손실함수, 옵티마이저, 메트릭스. 배치사이즈는 배수로 epochs는 횟수


# In[33]:


## 최소 제곱법으로 단항선형회귀분석(linear regression) 모델 생성
x = np.array([2,4,6,8])
y = np.array([81,93,91,97])


# In[38]:


mean_x = np.mean(x)  ## x의 평균
mean_y = np.mean(y)  ## y의 평균
print(f"x의평균값 :{mean_x}, y의평균값 : {mean_y}")


# In[44]:


# 기울기 공식 -> 합((x-x의평균)(y-y의평균))   /합 ((x-x의평균)의 제곱)
divisor =sum([(i-mean_x)**2 for i in x ])

top=0 
for i in range(len(x)):
    top +=(x[i]-mean_x)*(y[i]-mean_y)
    

    
a = top/divisor
print(a)
    


# In[46]:


## 절편 b = y-(기울기(a)*x)
b= mean_y - (a*mean_x)
print(b)


# In[50]:


pred_y = [a*i+b for i in x]

print(pred_y)


# In[64]:


## 간단한 그래프 그려보기


plt.figure(figsize=(5,5))

plt.plot(x,y,'o')
plt.plot(x,y)
plt.plot(x,pred_y,'o')
plt.plot(x,pred_y)
         
plt.show()


# In[73]:


## 평균 제곱 오차 (MSE) 방법  오차의 제곱의 평균이 0에 가깝도록 하는것이 목표 
fake_a = 3  #임의의 기울기 생성
fake_b = 76 ##임의의 절편 생성

# 공부 시간 과 성적  배열 생성
x = np.array([2,4,6,8])
y = np.array([81,93,91,97])


def predict(x):
    return x*fake_a +fake_b

predict_result =[predict(i)for i in x] ## x의 요소를 하나하나 predict함수에 넣어라   # 예측값 생성
print(predict_result)

mse_result = [sum((y[i]-predict_result[i])**2 for i in range(len(y)))/ len(y)]
print(mse_result)


# In[75]:


for x_value,y_value,y_predict in zip(x,y,predict_result):
    print(x_value,y_value,y_predict)


# In[88]:


# 경사하강법으로 a와 b를 검색 : MSE가 최소가 되도록 찾는다
x = np.array([2,4,6,8])
y = np.array([81,93,91,97])

a = 0  ## 기울기를 0으로 초기화
b= 0  ## 절편 0 으로 초기화
lr =0.03 ## 학습률
n = len(x)
epoch  = 2001  ## 반복 횟수 설정


## 경사 하강법 실행
for i in range(epoch):
    y_pred = a*x + b ## 예측값 구함
    error = y-y_pred ## 실제 값과 예측 값의 차이
    a_diff = (2/n)*sum(-x*(error)) ## a를 기준으로 mse공식을 편미분한 상태 (기울기를 기준으로)
    b_diff = (2/n)*sum(-(error))   ## b를 기준으로 mse공식을 편미분한 상태 (절편을 기준으로)
    a = a-lr*a_diff  ## 다음 a값 수정
    b = b -lr*b_diff ## 다음 b 값 수정
    
    if i %100 ==0 :
        print("epochs : {}, 기울기 :{}, 절편:{}".format(i,a,b))
    
#최종 a와 b를 적용하여   y의 예측값 구함
y_pred = a*x +b


plt.scatter(x,y)
plt.plot(x,y_pred, 'r')
plt.show()



