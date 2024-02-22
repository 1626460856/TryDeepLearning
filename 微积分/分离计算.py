from mxnet import autograd, np, npx

npx.set_np()
# 创建向量变量
x = np.arange(5.0)

# 创建 autograd 记录梯度
x.attach_grad()


with autograd.record():
    y = x * x
    #u = y.detach()
    z = y * x
z.backward()
print("这里返回的偏导数会经过y流向x，而不仅仅是z=y*x的式子里z关于x的偏导数")
print(x.grad)
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
print("这里返回的偏导数不会通过u流向x，这里中间变量u只是保留了y的值")
print(x.grad)