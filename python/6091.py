# %% 6091.py
a,b,c = input().split()
a= int(a)
b = int(b)
c = int(c)

d = 1
while d%a!=0 or d%b!=0 or d%c!=0 :
  d += 1
print(d)
# %% 6092.py
n = int(input())      #개수를 입력받아 n에 정수로 저장
a : list = input().split()  #공백을 기준으로 잘라 a에 순서대로 저장

for i in range(n) :  #0부터 n-1까지...
  a[i] = int(a[i])       #a에 순서대로 저장되어있는 각 값을 정수로 변환해 다시 저장

d = []                     #d라는 이름의 빈 리스트 [ ] 변수를 만듦. 대괄호 기호 [  ] 를 사용한다.
for i in range(24) :  #[0, 0, 0, ... , 0, 0, 0] 과 같이 24개의 정수 값 0을 추가해 넣음
  d.append(0)        #각 값은 d[0], d[1], d[2], ... , d[22], d[23] 으로 값을 읽고 저장할 수 있음.

for i in range(n) :    #번호를 부를 때마다, 그 번호에 대한 카운트 1씩 증가
  d[a[i]] += 1

for i in range(1, 24) :  #카운트한 값을 공백을 두고 출력
  print(d[i], end=' ')

# %% 6093.py
n = int(input())
a : list = input().split()

for i in range(n) :
    a[i] = int(a[i])

for i in range(n-1,-1,-1) :
    print(a[i],end=' ')
# %% 6094.py
n = int(input())
a : list = input().split()

for i in range(n) :
    a[i] = int(a[i])

min = 999999999999999999 # min=a[i]

for i in  range(n) : 
    if min > a[i] :
        min = a[i]
print(min)

# %% 6095.py
d=[]                        #대괄호 [ ] 를 이용해 아무것도 없는 빈 리스트 만들기
for i in range(20) :
  d.append([])         #리스트 안에 다른 리스트 추가해 넣기
  for j in range(20) : 
    d[i].append(0)    #리스트 안에 들어있는 리스트 안에 0 추가해 넣기

n = int(input())
for i in range(n) :
  x, y = input().split()
  d[int(x)][int(y)] = 1

for i in range(1, 20) :
  for j in range(1, 20) : 
    print(d[i][j], end=' ')    #공백을 두고 한 줄로 출력
  print()                          #줄 바꿈

# %% 6096.py
d=[]                        #대괄호 [ ] 를 이용해 아무것도 없는 빈 리스트 만들기
for i in range(19) :
    a = list(map(int, input().split()))
    d.append(a)         #리스트 안에 다른 리스트 추가해 넣기 
n= int(input())
for i in range(n) :
  x, y=input().split()
  for j in range(19) :
    if d[j][int(y)-1]==0 :
      d[j][int(y)-1]=1
    else :
      d[j][int(y)-1]=0

    if d[int(x)-1][j]==0 :
      d[int(x)-1][j]=1
    else :
      d[int(x)-1][j]=0
for i in range(19) :
  for j in range(19) : 
    print(d[i][j], end=' ')    #공백을 두고 한 줄로 출력
  print()                          #줄 바꿈
