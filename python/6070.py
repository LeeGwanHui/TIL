# %% 6070.py
month = int(input())
if month//3==1:
    print("spring")
elif month//3 ==2 :
    print("summer")
elif month//3 ==3 :
    print("fall")
else :
    print("winter")

# %% 6071.py
n = 1      #처음 조건 검사를 통과하기 위해 0 아닌 값을 임의로 저장
while n!=0 :
  n = int(input())
  if n!=0 :
    print(n)
# %% 6072.py
n = int(input())
while n!=0 :
  print(n)
  n = n-1

# %% 6073.py
n = int(input())
while n!=0 :
  n = n-1
  print(n)

# %% 6074.py
c = ord(input())
t = ord('a')
while t<=c :
  print(chr(t), end=' ') # end를 넣으므로써 한줄로 출력 가능
  t += 1

# %% 6075.py
n = int(input())
k=0
while k != n+1 :
    print(k)
    k+=1
# %% 6076.py
n = int(input())
for i in range(n+1) :
  print(i)

# %% 6077.py
n = int(input())
s = 0
for i in range(1, n+1) :
  if i%2==0 :
    s += i
print(s)
# %% 6078.py
n = '.'      #처음 조건 검사를 통과하기 위해 0 아닌 값을 임의로 저장
while n!='q' :
    n = input()
    print(n)
# %% 6079.py
n=0
sum = 0
req = int(input())
while sum<req :
    n = n+1
    sum = sum+n
print(n)
# %% 6070.py

n, m = input().split()
n = int(n)
m = int(m)

for i in range(1, n+1) :
  for j in range(1, m+1) :
    print(i, j)
# %%
