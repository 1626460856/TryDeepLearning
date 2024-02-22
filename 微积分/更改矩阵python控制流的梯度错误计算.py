from mxnet import autograd, np, npx

npx.set_np()


def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:  # 将向量b中的每个元素平方求和后再开平方根
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

#a = np.arange(5.0)
a = np.zeros((5, 4))
# 使用NumPy库中的random模块生成一个服从标准正态分布（均值为0，标准差为1）的随机数，并将其赋值给变量a
for i in range(5):
    for j in range(4):
        a[i][j] = np.random.normal()

print("随机数a：",a)
a.attach_grad()
with autograd.record():
    d = f(a)
    print("通过变换之后，会有某个值使得d=k*a，这里d的导数就是k,k=d/a：",d/a)
d.backward()

print("计算的d的导数是：",a.grad)
