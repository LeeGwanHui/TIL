##6042.py
a=float(input())
print( format(a, ".2f") )
##6043.py
a, b=input().split()
a= float(a)
b= float(b)
c= a/b
print( format(c, ".3f") )
## 6044.py
a,b = input().split()
a= int(a)
b= int(b)
print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)
print(format(a/b,".2f"))
## 6045.py
a,b,c = input().split()
a= int(a)
b= int(b)
c= int (c)
print(a+b+c, format((a+b+c)/3,".2f"))
## 6046.py
n = int(input())
print(n<<1)  #10을 2배 한 값인 20 이 출력된다.
## 6047.py
a,b = input().split()
a=int(a)
b=int(b)
c = a*(2**b)
#print(c)
print(a << b)  #210 = 1024 가 출력된다.
## 6048.py
a, b = input().split()
a = int(a)
b = int(b)
print(a<b)
## 6049.py
a, b = input().split()
a = int(a)
b = int(b)
print(a==b)
## 6050.py
a, b = input().split()
a = int(a)
b = int(b)
print(a<=b)
## 6051.py
a, b = input().split()
a = int(a)
b = int(b)
print(a!=b)
## 6052.py
n = int(input())
print(bool(n))
## 6053.py
a = bool(int(input()))
print(not a)
##6054.py
a, b = input().split()
print(bool(int(a)) and bool(int(b)))
##6055.py
a, b = input().split()
print(bool(int(a)) or bool(int(b)))
## 6056.py
a, b = input().split()
c = bool(int(a))
d = bool(int(b))
print((c and (not d)) or ((not c) and d))
## 6057.py
a, b = input().split()
c = bool(int(a))
d = bool(int(b))
print(((not c) and (not d)) or ( c and d))
##6058.py
a, b = input().split()
c = bool(int(a))
d = bool(int(b))
print(((not c) and (not d)))
##6059.py
a = int(input())
print(~a) #2진수 표현의 1의 보수를 나타냄
##6060.py
a, b = input().split()
a= int(a)
b= int(b)
print(a&b)
##6061.py
a, b = input().split()
a= int(a)
b= int(b)
print(a|b)
##6062.py
a, b = input().split()
a= int(a)
b= int(b)
print(a^b)