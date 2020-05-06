X = cPickle.load(open('./data_files/pca_spring.pkl','rb'))
X = np.asmatrix(X.T)

print(X.shape)
m = X.shape[0]

plt.figure(figsize = (12, 6))
plt.subplot(1,3,1)
plt.plot(X[:, 0], -X[:, 1], 'r')
plt.axis('equal')
plt.title('Camera 1')

plt.subplot(1,3,2)
plt.plot(X[:, 2], -X[:, 3], 'b')
plt.axis('equal')
plt.title('Camera 2')

plt.subplot(1,3,3)
plt.plot(X[:, 4], -X[:, 5], 'k')
plt.axis('equal')
plt.title('Camera 3')

plt.show()
X = X - np.mean(X, axis = 0)

S = 1/(m-1)*X.T*X

D, U = np.linalg.eig(S)

idx = np.argsort(-D)
D = D[idx]
U = U[:,idx]

print(D, '\n')
print(U)

plt.figure(figsize = (10,8))
plt.stem(np.sqrt(D))
plt.grid(alpha = 0.3)
plt.show()

# relative magnitutes of the principal components

Z = X*U
xp = np.arange(0, m)/24    # 24 frame rate

plt.figure(figsize = (10, 8))
plt.plot(xp, Z)
plt.yticks([])
plt.show()

## projected onto the first principal component
# 6 dim -> 1 dim (dim reduction)
# relative magnitute of the first principal component

Z = X*U[:,0]

plt.figure(figsize = (10, 8))
plt.plot(Z)
plt.yticks([])
plt.show()
