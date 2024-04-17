
import logging
import os
import open3d

def kdtree_partition(pc, max_num):
    parts = []

    class KD_node:
        def __init__(self, point=None, LL = None, RR = None):
            self.point = point
            self.left = LL
            self.right = RR

    def createKDTree(root, data):
        if len(data) <= max_num:
            parts.append(data)
            return

        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]

        point = data_sorted[int(len(data)/2)]

        root = KD_node(point)
        root.left = createKDTree(root.left, data_sorted[: int((len(data) / 2))])
        root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):])

        return root

    init_root = KD_node(None)
    _ = createKDTree(init_root, pc)
    return parts

def load_ply_data(filename):
    '''
    load data from ply file.
    '''
    f = open(filename)
    # 1.read all points
    points = []
    for line in f:
        # only x,y,z
        wordslist = line.split(' ')
        try:
            x, y, z = float(wordslist[0]), float(wordslist[1]), float(wordslist[2])
        except ValueError:
            continue
        points.append([x, y, z])
    points = np.array(points)
    points = points.astype(np.int32)  # np.uint8
    # print(filename,'\n','length:',points.shape)
    f.close()
    return points
def get_D2(filename):
    point_cloud = load_ply_data(filename)
    data = torch.from_numpy(point_cloud)
    # data = data.unsqueeze(0)
    data = torch.tensor(data).to(torch.float32)
    ori_pcd = open3d.geometry.PointCloud()  # 定义点云
    ori_pcd.points = open3d.utility.Vector3dVector(np.squeeze(data))  # 定义点云坐标位置[N,3]
    ori_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))  # 计算normal
    orifile = filename
    open3d.io.write_point_cloud(orifile, ori_pcd, write_ascii=True)
    # 将ply文件中normal类型double转为float32
    lines = open(orifile).readlines()
    to_be_modified = [4, 5, 6, 7, 8, 9]
    for i in to_be_modified:
        lines[i] = lines[i].replace('double', 'float')
    file = open(orifile, 'w')
    for line in lines:
        file.write(line)
    file.close()
def getlogger(logdir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger
import math
import os
import torch
import MinkowskiEngine as ME
import numpy as np
def get_points_number(filedir):
    if filedir.endswith('ply'):
        plyfile = open(filedir)
        line = plyfile.readline()
        while line.find("element vertex") == -1:
            line = plyfile.readline()
        number = int(line.split(' ')[-1][:-1])
    elif filedir.endswith("bin"):
        number = len(np.fromfile(filedir, dtype='float32').reshape(-1, 4))
    elif filedir.endswith("h5"):
        file = h5py.File(filedir, 'r')
        number = int(file['data'].shape[0])

    return number
def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    coords = coords.astype('float')
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
    f.close()

    return
def sort_sparse_tensor(sparse_tensor):
    """ Sort points in sparse tensor according to their coordinates.
    """
    indices = torch.argsort(array2vector(sparse_tensor.C,
                                           sparse_tensor.C.max()+1))
    sparse_tensor = create_new_sparse_tensor(coordinates=sparse_tensor.C[indices],
                                            features=sparse_tensor.F[indices],
                                            tensor_stride=sparse_tensor.tensor_stride,
                                            dimension=sparse_tensor.D,
                                            device=sparse_tensor.device)

    return sparse_tensor

def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long(), step.long()
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector
def create_new_sparse_tensor(coordinates, features, tensor_stride, dimension, device):
    manager = ME.CoordinateManager(D=dimension)
    key, _ = manager.insert_and_map(coordinates.to(device), tensor_stride)
    sparse_tensor = ME.SparseTensor(features=features,
                                    coordinate_map_key=key,
                                    coordinate_manager=manager,
                                    device=device)

    return sparse_tensor


def topk_1(out_cls, out, num_points):
    prob = torch.sigmoid(out_cls.F)
    mask = istopk_local(prob, k=1)
    prob[torch.where(mask)[0]] = 1
    mask = istopk_global(prob, k=num_points)
    prun = ME.MinkowskiPruning()
    out = prun(out, mask.to(out.device))

    return out


def istopk(data, nums, rho=1.0):
    """ Input data is sparse tensor and nums is a list of shape [batch_size].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N * rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)  # must CPU.
        mask[row_indices[indices]] = True

    return mask.bool().to(data.device)


def istopk_local(data, k=1):
    """input data is probability
    select top-k voxels in each 8-voxels set
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    N=len(data)//8
    data=data[:N*8]
    _, indices = torch.topk(data.reshape(-1, 8), k)
    indices += (torch.arange(0, len(indices)) * 8).reshape(-1, 1).to(indices.device)
    indices = indices.reshape(-1)
    mask[indices] = True

    return mask.bool().to(data.device)
def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long().cpu(), step.long().cpu()
    vector = sum([array[:, i] * (step ** i) for i in range(array.shape[-1])])

    return vector
def scale_sparse_tensor(x, factor, quant_mode='round'):
    if factor==1: return x
    assert quant_mode=='floor' or quant_mode=='round'
    coords = x.C.cpu().clone().float()
    coords[:,1:] = coords[:,1:]*factor
    if quant_mode=='round':
        coords[:,1:] = torch.round(coords[:,1:]).int()
    elif quant_mode=='floor':
        coords[:,1:] = torch.floor(coords[:,1:]).int()
    coords = torch.unique(coords, dim=0).int()
    feats = torch.ones((len(coords),1)).float()
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=x.device)

    return x

def istopk_global(data, k):
    """input data is probability
    select top-k voxel in all voxels
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    _, indices = torch.topk(data.squeeze(), k)
    mask[indices] = True

    return mask.bool().to(data.device)

def read_ply_ascii_geo(filedir):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:, 0:3].astype('int')
    # coords = data[:,0:3].astype('float')

    return coords
#
# def judge_density(coords,QP):
#     res=math.log2(np.max(coords))
#     Rate,R,k=choice_rate(res,QP)
#     dist, num = get_local_density(torch.tensor(coords).unsqueeze(0).cuda().float(), density_radius=R,k=k)
#     volume = 4 / 3 * math.pi * R ** 3
#     IQR = num / volume
#     # print(IQR)
#     IQR = torch.sort(IQR, descending=True)[0]
#     print('median-----')
#     print("%.7f" % torch.median(IQR))
#     # print('mean-----')
#     # print("%.7f" % torch.mean(IQR))
#     # print('max-----')
#     # print("%.7f" % torch.max(IQR))
#     # print('min-----')
#     # print("%.7f" % torch.min(IQR))
#     result = choice_pc(torch.median(IQR),Rate)
#     return result
# def choice_pc(IQR,Rate):
#     if Rate == 1:
#         if IQR>=2e-4:
#             result ='solid'
#         elif IQR>=1.5e-5:
#             result ='dense'
#         elif IQR<=1e-5:
#             result ='sparse'
#     if Rate == 2:
#         if IQR>=2e-3:
#             result ='solid'
#         elif IQR>=1e-4:
#             result ='dense'
#         elif IQR<=1e-4:
#             result ='sparse'
#     if Rate == 3:
#         if IQR>=1e-2:
#             result ='solid'
#         elif IQR>=2e-3:
#             result ='dense'
#         elif IQR<=2e-3:
#             result ='sparse'
#     if Rate == 4:
#         if IQR>=1e-2:
#             result ='solid'
#         elif IQR>=5e-3:
#             result ='dense'
#         elif IQR<=5e-3:
#             result ='sparse'
#     if Rate == 5:
#         # if IQR>=1e-2:
#         #     result ='solid'
#         if IQR>=1e-2:
#             result ='sparse'
#         elif IQR<=1e-2:
#             result ='sparse'
#     return result
# def choice_rate(res,QP):
#
#     if res==10:
#         if QP==0.125:
#             Rate =1
#             R=64
#             k=1000
#         if QP==0.25:
#             Rate = 2
#             R = 32
#             k = 400
#         if QP==0.5:
#             Rate = 3
#             R = 8
#             k = 100
#         if QP==0.75:
#             Rate = 4
#             R = 8
#             k = 100
#         if QP==0.875:
#             Rate = 5
#             R = 8
#             k = 100
#     elif res==11:
#         if QP==0.0625:
#             Rate = 1
#             R = 64
#             k = 1000
#         if QP==0.125:
#             Rate = 2
#             R = 32
#             k = 400
#         if QP==0.25:
#             Rate = 3
#             R = 8
#             k = 100
#         if QP==0.5:
#             Rate = 4
#             R = 8
#             k = 100
#         if QP==0.75:
#             Rate = 5
#             R = 8
#             k = 100
#     elif res==12:
#         if QP==0.03125:
#             Rate = 1
#             R = 64
#             k = 1000
#         if QP==0.0625:
#             Rate = 2
#             R = 32
#             k = 400
#         if QP==0.125:
#             Rate = 3
#             R = 8
#             k = 100
#         if QP==0.25:
#             Rate = 4
#             R = 8
#             k = 100
#         if QP==0.5:
#             Rate = 5
#             R = 8
#             k = 100
#     return Rate,R,k