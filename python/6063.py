
# %% 6063.py
a, b = input().split()
a = int(a)  #변수 a에 저장되어있는 값을 정수로 바꾸어 다시 변수 a에 저장
b = int(b)
c = (a if (a>=b) else b)
#print(type(c))
print(c)
# %% 6064.py
a, b, c = input().split()
a = int(a)  #변수 a에 저장되어있는 값을 정수로 바꾸어 다시 변수 a에 저장
b = int(b)
c = int(c)
k=(a if a<b else b) if ((a if a<b else b)<c) else c
print(k)
# %% 6065.py
a, b, c = input().split()

a = int(a)
b = int(b)
c = int(c)

if a%2==0 :  #논리적으로 한 단위로 처리해야하는 경우 콜론(:)을 찍고, 들여쓰기로 작성 한다.
  print(a)

if b%2==0 :
  print(b) 

if c%2==0 :
  print(c) 
# %% 6066.py
a, b, c = input().split()

a = int(a)
b = int(b)
c = int(c)

if a%2==0 :  #논리적으로 한 단위로 처리해야하는 경우 콜론(:)을 찍고, 들여쓰기로 작성 한다.
  print("even")

else : 
    print("odd")

if b%2==0 :  #논리적으로 한 단위로 처리해야하는 경우 콜론(:)을 찍고, 들여쓰기로 작성 한다.
  print("even")

else : 
    print("odd")
if c%2==0 :  #논리적으로 한 단위로 처리해야하는 경우 콜론(:)을 찍고, 들여쓰기로 작성 한다.
  print("even")

else : 
    print("odd")

# %% 6067.py
n = int(input())
if n<0 :
  if n%2==0 :
    print('A')      #주의 : 변수 A와 문자열 'A' / "A" 는 의미가 완전히 다르다. 
  else :
    print('B')
else :
  if n%2==0 :
    print('C')
  else :
    print('D')
#%% 6068.py
n = input()
if n =='A' :
  print('best!!!')
elif n == 'B' :
    print('good!!')
elif n == 'C' :
    print('run!')
elif n == 'D' :
    print('slowly~')
else :
    print("what?")
# %%
