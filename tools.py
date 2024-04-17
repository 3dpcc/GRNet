import subprocess
import time
from util import write_ply_ascii_geo
import os; rootdir = os.path.split(__file__)[0]
import numpy as np
# rootdir = './extension'

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try: number = float(item) 
        except ValueError: continue
        
    return number

def gpcc_encode(filedir, bin_dir, posQuantscale=1, show=False):
    """Compress point cloud geometry losslessly using MPEG G-PCCv14. 
    You can download and install TMC13 from https://github.com/MPEGGroup/mpeg-pcc-tmc13
    """
    # print('GPCC Encoding \t......')
    command=' --trisoupNodeSizeLog2=0' + \
            ' --neighbourAvailBoundaryLog2=8' + \
            ' --intra_pred_max_node_size_log2=6' + \
            ' --maxNumQtBtBeforeOt=4' + \
            ' --mergeDuplicatedPoints=1' + \
            ' --planarEnabled=1' + \
            ' --planarModeIdcmUse=0' + \
            ' --minQtbtSizeLog2=0'
    if posQuantscale==1: command+=' --inferredDirectCodingMode=1'

    subp=subprocess.Popen(rootdir+'/tmc3 --mode=0' + command + \
                        ' --positionQuantizationScale='+str(posQuantscale) + \
                        ' --uncompressedDataPath='+filedir + \
                        ' --compressedStreamPath='+str(bin_dir),
                        shell=True, stdout=subprocess.PIPE)
    headers = ['Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    results = {}
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value
        c=subp.stdout.readline()
    # print('Encoding Done.', '\tTime:', round(results['Processing time (wall)'], 3), 's')
    
    return results
def gpcc_pred_encode(filedir, bin_dir, posQuantscale=1, show=False):
    """Compress point cloud geometry losslessly using MPEG G-PCCv14. 
    You can download and install TMC13 from https://github.com/MPEGGroup/mpeg-pcc-tmc13
    """
    # print('GPCC Encoding \t......')
    command=' --trisoupNodeSizeLog2=0' + \
            ' --neighbourAvailBoundaryLog2=8' + \
            ' --intra_pred_max_node_size_log2=6' + \
            ' --maxNumQtBtBeforeOt=4' + \
            ' --mergeDuplicatedPoints=1' + \
            ' --planarEnabled=1' + \
            ' --planarModeIdcmUse=0' + \
            ' --minQtbtSizeLog2=0'
    if posQuantscale==1: command+=' --inferredDirectCodingMode=1'
    subp=subprocess.Popen(rootdir+'/tmc3 --mode=0' + command + \
                        ' --positionQuantizationScale='+str(posQuantscale) + \
                        ' --uncompressedDataPath='+filedir + \
                        ' --compressedStreamPath='+bin_dir, 
                        shell=True, stdout=subprocess.PIPE)
    headers = ['Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    results = {}
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value
        c=subp.stdout.readline()
    # print('Encoding Done.', '\tTime:', round(results['Processing time (wall)'], 3), 's')
    
    return results
def gpcc_pred_decode(bin_dir, dec_dir, show=False):
    # print('Decoding \t......')
    subp=subprocess.Popen(rootdir+'/tmc3 --mode=1'+  
                            ' --compressedStreamPath='+bin_dir+ 
                            ' --reconstructedDataPath='+dec_dir+
                            ' --outputBinaryPly=0',
                            shell=True, stdout=subprocess.PIPE)
    headers = ['Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    results = {}
    c=subp.stdout.readline()
    while c:
        if show: print(c)   
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value   
        c=subp.stdout.readline()
    # print('Decoding Done.', '\tTime:', round(results['Processing time (wall)'], 3), 's')

    return results
def gpcc_decode(bin_dir, dec_dir, show=False):
    # print('Decoding \t......')
    subp=subprocess.Popen(rootdir+'/tmc3 --mode=1'+  
                            ' --compressedStreamPath='+bin_dir+ 
                            ' --reconstructedDataPath='+dec_dir+
                            ' --outputBinaryPly=0',
                            shell=True, stdout=subprocess.PIPE)
    headers = ['Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    results = {}
    c=subp.stdout.readline()
    while c:
        if show: print(c)   
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value   
        c=subp.stdout.readline()
    # print('Decoding Done.', '\tTime:', round(results['Processing time (wall)'], 3), 's')

    return results

def pc_error(infile1, infile2, resolution, normal=False, show=False):
    # print('Test distortion\t......')
    # print('resolution:\t', resolution)
    start_time = time.time()
    # headersF = ["mse1      (p2point)", "mse1,PSNR (p2point)",
    #            "h.       1(p2point)", "h.,PSNR  1(p2point)",
    #            "mse2      (p2point)", "mse2,PSNR (p2point)",
    #            "h.       2(p2point)", "h.,PSNR  2(p2point)" ,
    #            "mseF      (p2point)", "mseF,PSNR (p2point)",
    #            "h.        (p2point)", "h.,PSNR   (p2point)" ]
    # headersF_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
    #                   "mse2      (p2plane)", "mse2,PSNR (p2plane)",
    #                   "mseF      (p2plane)", "mseF,PSNR (p2plane)"]
    headers = ["mseF      (p2point)", "mseF,PSNR (p2point)"]
    command = str(rootdir+'/pc_error_d' +
                          ' -a '+infile1+
                          ' -b '+infile2+
                          ' --hausdorff=1 '+
                          ' --resolution='+str(resolution))
    if normal:
        headers +=["mseF      (p2plane)", "mseF,PSNR (p2plane)"]
        command = str(command + ' -n ' + infile1)

    results = {}
    subp=subprocess.Popen(command,shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')# python3.
        if show: print(line)
        for _, key in enumerate(headers):

            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value
        c=subp.stdout.readline()
    # print('Test Distortion Done.', '\tTime:', round(time.time() - start_time, 3), 's'
    return results


class CoordinateCoder():
    """write coordinates to file.
    """

    def encode(self, coords, filename):
        coords = coords.astype('int')
        with open(filename + '_C.bin', 'wb') as fout:
            min_coords = coords.min(axis=0)

            write_ply_ascii_geo("./longdress_vox10_1300_down.ply", coords)

            coords -= min_coords

            assert coords.max() <= 255
            fout.write(np.array(min_coords, dtype=np.int16).tobytes())
            fout.write(np.array(coords, dtype=np.uint8).tobytes())

        with open(filename + '_C.bin', 'rb') as fin:
            min_coords_1 = np.frombuffer(fin.read(3 * 2), dtype=np.int16).reshape(-1, 3)
            coords_1 = np.frombuffer(fin.read(), dtype=np.uint8).reshape(-1, 3)
            coords_dec = coords_1 + min_coords_1
        # print(coords_dec)
        # print(coords)

        assert (coords.sort() == coords_dec.sort())
        # with open(filename + '_C.bin', 'rb') as fin:
        #     print(fin.read())

        return

    def decode(self, filename):
        with open(filename + '_C.bin', 'rb') as fin:
            min_coords = np.frombuffer(fin.read(3 * 2), dtype=np.int16).reshape(-1, 3)
            coords = np.frombuffer(fin.read(), dtype=np.uint8).reshape(-1, 3)

            # print(min_coords)
            # print(coords)

            coords = coords + min_coords

        return coords

