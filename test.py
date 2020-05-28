import  numpy as np
import  collections
import scipy.sparse as sp
import  collections
import  random as rd
import  tensorflow as tf

row  = np.array([1, 2, 3, 4])
col  = np.array([0, 3, 1, 2])
data = np.array([10, 11, 12, 13])
A=sp.coo_matrix((data, (row, col)), shape=(6, 6))
B=sp.coo_matrix((data, (col, row)), shape=(6, 6))
C=[]
C.append(A)
C.append(B)
M=sum(C)
print(M.toarray())
print(M)
print("______________________")
print(M[1])
print("************************")
print(M[0:3])
print("___________________")
# coo = M[0:4].tocoo().astype(np.float32)
# indices = np.mat([coo.row, coo.col]).transpose()
# print(coo)
# print(coo.data,coo.shape)
# for i in range(100):
#     print(i*100)
#     print((i+1)*100)
#     break
# adj_mat_list = []
# adj_mat_list.append(A)
# adj_mat_list.append(B)
# all_h_list=[]
# for id ,ll in enumerate(adj_mat_list):
#     print(ll.data)
#     all_h_list += list(ll.row)
#     print(all_h_list)
# print(iter)
# print(user)
# print(iter[:, 1])
# class load_kg():
#     def _load_kg( file_name):
#         def _construct_kg(kg_np):
#             kg = collections.defaultdict(list)
#             rd = collections.defaultdict(list)
#
#             for head, relation, tail in kg_np:
#                 kg[head].append((tail, relation))
#                 rd[relation].append((head, tail))
#             return kg, rd
#
#         kg_np = np.loadtxt(file_name, dtype=np.int32)
#         kg_np = np.unique(kg_np, axis=0)
#         print("kg_np",kg_np)
#         # self.n_relations = len(set(kg_np[:, 1]))
#         # self.n_entities = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
#         n_relations = max(kg_np[:, 1]) + 1
#         n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
#         n_triples = len(kg_np)
#
#         kg_dict, relation_dict = _construct_kg(kg_np)
#
#         return kg_np, kg_dict, relation_dict
# A,B,C=load_kg._load_kg("kg_test.txt")
# print(A,B,C)
# lines = open("kg_final.txt", 'r').readlines()
# x=0
# relations=set()
# print(relations)
# for l in lines:
#     tmps = l.strip()
#     inters = [int(i) for i in tmps.split(' ')]
#     relation=inters[0]
#     relations.add(relation)
# print(relations)
# M=dict()
# M[1] =[2,3,4]
# M[5]=[6,7,8]
# od = collections.OrderedDict(sorted(M.items()))
# print(od)
# for h, vals in od.items():
#     print(h)
#     print(vals)
# C=np.array([8,7,6])
# D=np.argsort(C)
# new_list = np.array(C)
# new_list = new_list[D]
# print(new_list)
# users_dict=dict()
# users_dict["A"]=[3]
# users_dict["B"]=[4]
# C=users_dict.keys()
# print(rd.random(C,1))
# np.random.seed(1)
# print(np.random.randint(low=0, high=5, size=1)[0])
# weight_size_list=[512]+[64,32,16]
# print(weight_size_list)

'''
[[1,0],
 [0,1]]
'''
# st = tf.SparseTensor(values=[4, 5,6], indices=[[0, 1], [1, 1],[1,2]], dense_shape=[3, 3])
# dt = tf.ones(shape=[3,3],dtype=tf.int32)
# result = tf.sparse_tensor_dense_matmul(st,dt)
# sess = tf.Session()
# with sess.as_default(): print(result.eval())
# C=[]
# A=[[1,0],[0,1]]
# B=[[1,0],[0,1]]
# C.append(A)
# C.append(B)
# result=tf.concat(C,axis=0)
# sess=tf.Session()
# print(sess.run(result))
