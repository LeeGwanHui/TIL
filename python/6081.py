# %% 6081.py
import imp


a=int(input(),16)
for i in range(1,16) :
  print('%X'%a, '*%X'%i, '=%X'%(a*i), sep='')
    #print(f'{hex(a)} x {hex(i)} = {hex(i*a)}')
# %% 6082.py
n = int(input())
for i in range(1, n+1) :
    if i%10==3 or i%10==6 or i%10==9 :
        print("X", end=' ')    #출력 후 공백문자(빈칸, ' ')로 끝냄
    else :
        print(i,end=' ')
# %% 6083.py
r, g, b = input().split()
r = int(r)
g = int(g)
b = int(b)
n=0
for i in range(r) :
    for j in range(g) :
        for k in range(b) :
            print(f'{i} {j} {k}')
            n+=1
print(f'{n}')
# %% 6084.py 
h, b, c, s = input().split()
h = int(h)
b = int(b)
c = int(c)
s = int(s)
storage = round(h*b*c*s/8/1024/1024,1)
print(f'{storage} MB')


# %% 6085.py
w, h, b = input().split()
w = int(w)
h = int(h)
b = int(b)
storage = round(w*h*b/8/1024/1024,2)
print(f'{storage:.2f} MB')

# %% 6086.py
num = int(input())
s=0
c=1
while True :
    s+=c
    c+=1
    if s>=num:
        print(s)
        break


# %% 6087.py
num = int(input())

for i in range(1,num+1):
    if i%3==0:
        continue
    print(i,end=' ')

# %% 6088.py
a, d, n = input().split()
a= int(a)
d= int(d)
n= int(n)

number = a
for i in range(1,n+1):
    if i==1:
        number = a
    else :
        number+=d
print(number)

# %% 6089.py
a, d, n = input().split()
a= int(a)
d= int(d)
n= int(n)

number = a
for i in range(2,n+1):
    number*=d
print(number)

# %% 6090.py
a, m, d, n = input().split()
a= int(a)
m = int(m)
d= int(d)
n= int(n)

number = a
for i in range(2,n+1):
    number= number*m +d
print(number)
# %%
