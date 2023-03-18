from annoy import AnnoyIndex
import random

f = 4  # Length of item vector that will be indexed

t = AnnoyIndex(f, 'angular')
for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)

t.add_item(1000, [1.2, 2.2, 3.2, 4.2])
t.add_item(1001, [1.3, 2.2, 3.2, 4.2])
t.add_item(1002, [1.3, 2.2, 3.2, 4.2])
t.add_item(1003, [1.3, 2.2, 3.2, 4.2])

t.build(10)  # 10 trees
t.save('test.ann')

# ...

u = AnnoyIndex(f, 'angular')
u.load('test.ann')  # super fast, will just mmap the file
vec=[1.3, 2.2, 3.2, 4.2]
id,value=u.get_nns_by_vector(vec, 1,include_distances=True)
print(id[0])  # will find the 1000 nearest neighbors
