# %% 6097.PY
h, w = input().split()
h = int(h)
w = int(w)

n = int(input())

d = []
for i in range(h):
    d.append([])
    for j in range(w) :
        d[i].append(0)


for i in range(n) :
    l, p, x, y =  map(int, input().split())
    for j in range(1):
        if p == 0 :
            for k in range(l):
                d[x-1][y-1+k] = 1
        if p == 1 :
            for k in range(l):
                d[x-1+k][y-1] = 1

for i in range(h) :
  for j in range(w) : 
    print(d[i][j], end=' ')    #공백을 두고 한 줄로 출력
  print()    

# %% 6098.py

# d = [
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
#     [1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
#     [1, 0, 0, 0, 0, 1, 2, 1, 0, 1],
#     [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#     ]

d=[]                        #대괄호 [ ] 를 이용해 아무것도 없는 빈 리스트 만들기
for i in range(10) :
    a = list(map(int, input().split()))
    d.append(a)   

x= 2-1
y= 2-1
d[y][x] = 9
while True : 
    if d[y][x+1] ==0 :
        d[y][x+1] = 9
        x +=1
    elif d[y][x+1] == 1 :
        if d[y+1][x] == 0 :
            d[y+1][x] = 9
            y+=1
        elif d[y+1][x] ==2 :
             d[y+1][x] = 9
             break
        else :
            break
    else : # d == 2
        d[y][x+1] = 9
        break


for i in range(10) :
  for j in range(10) : 
    print(d[i][j], end=' ')    #공백을 두고 한 줄로 출력
  print()    


# %%
