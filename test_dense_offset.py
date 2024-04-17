import open3d as o3d
import os, glob, argparse
import numpy as np
import torch
import MinkowskiEngine as ME
import time
from model.GRNet_dense_offset import knn_multiscale
from util import kdtree_partition,write_ply_ascii_geo,read_ply_ascii_geo,get_points_number
from tool.pc_error import pc_error
from tools import gpcc_encode,gpcc_decode
import pandas as pd

def parse_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--ckpts", default='./ckpts/dense_offset/r04/last_epoch.pth', help='Path to pretrained model')
  parser.add_argument("--output_dir", type=str, default='./output/dense/', help="Output dir")
  parser.add_argument("--gpcc_outdir", type=str, default='./gpcc_out/')
  parser.add_argument("--rate_dir", type=str, default='r04/', help="G-PCC(octree) rate point.")
  parser.add_argument("--GT_dir",default='/media/ivc-18/disk/testdata/GRNet_GT_12bit_dense/',help='Ground truth point cloud dir')
  parser.add_argument("--resolution", type=int,default=4095, help='Follow MPEG CTC ,9bit:511, 10bit:1023, 11bit:2047, 12bit:4095')
  parser.add_argument("--posQuantscale", type=float, default=0.25,  help='PosQuantscale of G-PCC(octree), R01:0.03125, R02:0.0625,'
                                                                         ' R03:0.125, R04:0.25, R05:0.5')
  parser.add_argument("--max_nums", type=int, default=100000, help='Max points number for kd-tree partition')
  args = parser.parse_args()
  return args

args = parse_args()
GT_files = glob.glob(args.GT_dir+'*.ply')
GT_files=sorted(GT_files)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = knn_multiscale().to(device)
ckpt = torch.load(args.ckpts,map_location='cuda:0')
max_nums=args.max_nums
model.load_state_dict(ckpt['model'])
gpcc_outdir =args.gpcc_outdir
posQuantscale=args.posQuantscale
l = len(GT_files)
save_path = os.path.join(args.output_dir, args.rate_dir)
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(gpcc_outdir):
    os.makedirs(gpcc_outdir)
i=0

for  pc_gt in GT_files:
    print("################## FILE NUMBER#################: ", i, ' / ', l)
    print('model parameter:', sum(param.numel() for param in model.parameters()))
    ####################### Read point cloud ###########
    file_name = os.path.basename(pc_gt)
    pc_name=file_name.split('.')[0]
    pcd = o3d.io.read_point_cloud(pc_gt)
    pcd_gt = o3d.io.read_point_cloud(pc_gt)
    coords_gt = np.asarray(pcd_gt.points)
    num_gt =len(coords_gt)
    ####################################################

    ####################### Partition ##################
    parts_gt = kdtree_partition(coords_gt,max_nums)
    ####################################################

    bits_coordinates = 0
    run_time=0
    out_list = []
    gpcc_list=[]

    for j,part_gt in enumerate(parts_gt):
        partdir='./gpcc_out/test'+'part'+str(j)+'.ply'
        write_ply_ascii_geo(partdir,part_gt)
        part_T = ME.utils.batched_coordinates([part_gt])

        ####################### G-PCC(octree) lossy compression ####################
        bin_dir = gpcc_outdir + 'part'+str(j) + '.bin'
        dec_dir = gpcc_outdir + 'part'+str(j) + '_dec.ply'

        results_enc = gpcc_encode(partdir, bin_dir, posQuantscale=posQuantscale)
        results_dec = gpcc_decode(bin_dir, dec_dir)
        # bpp
        num_points = get_points_number(dec_dir)
        bpp = round(8 * results_enc['Total bitstream size'] / get_points_number(partdir), 4)
        bits_coordinates = 8*results_enc['Total bitstream size']+bits_coordinates
        # results
        results_gpcc = {'posQuantscale': posQuantscale, 'num_points': num_points, 'bpp': bpp,
                   'bytes': int(results_enc['Total bitstream size']),
                   'time (enc)': results_enc['Processing time (user)'],
                   'time (dec)': results_dec['Processing time (user)']}
        part_dec=read_ply_ascii_geo(dec_dir)
        #########################################################################

        start_time = time.time()
        ####################### Voxelization #############
        c_dec = ME.utils.batched_coordinates([part_dec])
        f = torch.from_numpy(np.vstack(np.expand_dims(np.ones(c_dec.shape[0]), 1))).float()
        x = ME.SparseTensor(features=f, coordinates=c_dec,device=device)
        p2 = ME.utils.batched_coordinates([part_T])
        ##################################################

        ####################### GRNet ####################
        with torch.no_grad():
            out_C= model(x)
        ##################################################
        run_time += time.time() - start_time
        gpcc_list.append(x.C[:, 1:])

        out_list.append(torch.round(out_C[0].squeeze(0)))

    print("Point Cloud:",file_name)
    print("bpp_coordinates:",bits_coordinates/num_gt)
    run_time = round(run_time, 3)
    rec_pc = torch.cat(out_list, 0)
    gpcc_pc = torch.cat(gpcc_list, 0)

    print("Number of points in G-PCC compressed point cloud : ", gpcc_pc.shape[0])
    print("Number of points in GRNet restorated point cloud : ", rec_pc.cpu().numpy().shape[0])
    rec_pcd = o3d.geometry.PointCloud()
    recfile = os.path.join(save_path, file_name)
    write_ply_ascii_geo(recfile, rec_pc.detach().cpu().numpy())
    write_ply_ascii_geo(recfile + 'gpcc.ply', gpcc_pc.detach().cpu().numpy())
    print('Run Time:\t', run_time, 's')
    GT_dirs=args.GT_dir
    pc_error_metrics = pc_error(GT_dirs+file_name, recfile, res=args.resolution, show=False)
    pc_error_metrics_gpcc=pc_error(GT_dirs+file_name, recfile + 'gpcc.ply', res=args.resolution, show=False)

    ###########################Results#############################
    results = {}
    print('----------------GRNet PSNR D1-------------------')
    print(pc_error_metrics["mseF,PSNR (p2point)"][0])
    print('----------------GRNet PSNR D2-------------------')
    print(pc_error_metrics["mseF,PSNR (p2plane)"][0])

    print('----------------G-PCC PSNR D1-------------------')
    print(pc_error_metrics_gpcc["mseF,PSNR (p2point)"][0])
    print('----------------G-PCC PSNR D2-------------------')
    print(pc_error_metrics_gpcc["mseF,PSNR (p2plane)"][0])

    results["filename"] = file_name
    results["bpp(coords)"] = bits_coordinates / num_gt
    results['GRNet PSNR D1'] = pc_error_metrics["mseF,PSNR (p2point)"][0]
    results['GRNet PSNR D2'] = pc_error_metrics["mseF,PSNR (p2plane)"][0]
    results["run time"] = run_time
    results['G-PCC PSNR D1'] = pc_error_metrics_gpcc["mseF,PSNR (p2point)"][0]
    results['G-PCC PSNR D2'] = pc_error_metrics_gpcc["mseF,PSNR (p2plane)"][0]
    results["gpcc enc time"] = results_gpcc['time (enc)']
    results["gpcc dec time"] = results_gpcc['time (dec)']
    ################################################################

    ########################### Write to excel #####################
    csv_name = os.path.join(save_path, 'all_result'+str(args.resolution)  + '.csv')
    results = pd.DataFrame([results])
    if i == 0:
        results_allfile = results.copy(deep=True)
    else:
        results_allfile = results_allfile.append(results, ignore_index=True)
    csvfile = os.path.join(save_path, 'results'+str(args.resolution) + '.csv')
    results_allfile.to_csv(csv_name, index=False)
    print('Wrile results to: \t', csv_name)
    i += 1
    torch.cuda.empty_cache()
    ################################################################


