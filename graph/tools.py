import numpy as np

def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A



def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_R(num_node,joint_part_body_link):
    # 首先创建一个大小为(25,num_node)的全零矩阵
    R = np.zeros((25,num_node))
    # 使用嵌套的 for 循环遍历 joint_part_body_link 列表中的每个元素。
    # joint_part_body_link 是一个二维列表，每个子列表表示一个身体部位，包含了该部位涉及的节点索引。
    # 在循环中，将 R 中对应节点索引和身体部位的位置设置为 1，表示该节点属于该身体部位。
    for i in range(len(joint_part_body_link)):
        for j in range(len(joint_part_body_link[i])):
            R[joint_part_body_link[i][j],i] = 1
    # print("R:------------")
    # print(R)
    return R

def get_left(num_node,self_link, inward, outward,part_item):
    # 首先创建一个空列表 A_cross_level，用于存储计算得到的连接矩阵
    A_cross_level = []
    # I = edge2mat(self_link,num_node)
    In = normalize_digraph(edge2mat(inward,num_node))
    Out = normalize_digraph(edge2mat(outward,num_node))
    A = np.stack((In,Out)) # 获得身体部位的自身连接，以及每个身体部位的相邻连接
    R = get_R(num_node,part_item) # 获得每个节点是属于哪个身体部位的
    for i in range(len(A)):
        A_cross_level.append(R @ A[i] @ np.transpose(R,axes=[1,0]))
    # print("A_cross_level:------------")
    # print(A_cross_level)
    # 返回cross_level[1]是指左侧连接的矩阵   cross_level[0]是指自连接的矩阵
    return A_cross_level[1]