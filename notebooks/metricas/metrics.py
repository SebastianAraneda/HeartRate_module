import pyVHR as vhr
import numpy as np
import pandas as pd
import os

from pyVHR.analysis.pipeline import Pipeline

dataset_name = 'lgi_ppgi'          
video_DIR = '../datasets/LGI-PPGI/' 
BVP_DIR = '../datasets/LGI-PPGI/'    # dir containing BVPs GT

dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR= video_DIR, BVPdataDIR= BVP_DIR)
allvideo = dataset.videoFilenames
allground = dataset.sigFilenames
pipeline = Pipeline()

titles = []
for i in allvideo:
    txt = i.split('/')[4]
    txt_ = txt.split('_')
    titles.append(( txt_[0],txt_[1]))
# get the name of i video: titles[i][0]
# get the activity of i video: tittles[i][1]

def get_video_data(met, app, video_index):
    BPMs_predicted = []
    times_predicted = []
    uncerts_predicted = []

    bpmgt, timegt = vhr.datasets.lgi_ppgi.LGI_PPGI(videodataDIR= allvideo[video_index], BVPdataDIR= allground[video_index]).readSigfile(filename= allground[video_index]).getBPM()
    timegt = [int(item) for item in timegt]

    for i in allvideo:
        txt = i.split('/')[4]
        txt_ = txt.split('_')
        title = ( txt_[0],txt_[1]) 

    methods = ['cupy_POS', 'torch_CHROM', 'cupy_CHROM']

    for method in methods:
        timepd, bpmpd, uncertpd = pipeline.run_on_video(allvideo[video_index],cuda = True, roi_method=met, roi_approach=app, method= method,  verb=False )
        BPMs_predicted.append(bpmpd)
        times_predicted.append(timepd)
        uncerts_predicted.append(uncertpd)

    data_cupy_POS = {
        'time': times_predicted[0], 
        'BPM predicted': BPMs_predicted[0],
        'uncert': uncerts_predicted[0],
        'rPPG method' : methods[0],
        }
    data_torch_CHROM = {
        'time': times_predicted[0], 
        'BPM predicted': BPMs_predicted[1],
        'uncert': uncerts_predicted[1],
        'rPPG method' : methods[1]
        }
    data_cupy_CHROM = {
        'time': times_predicted[2], 
        'BPM predicted': BPMs_predicted[2],
        'uncert': uncerts_predicted[2],
        'rPPG method' : methods[2]
        }
    data_gt = {'extraction': met, 'approach': app, 'BPM gt': bpmgt, 'time gt': timegt}
    
    return(data_cupy_CHROM, data_torch_CHROM, data_cupy_POS, data_gt)

def get_video_dataframes(data_cupy_CHROM, data_torch_CHROM, data_cupy_POS, data_gt):
    frame_cupy_POS = pd.DataFrame(data_cupy_POS)
    frame_torch_POS = pd.DataFrame(data_torch_CHROM)
    frame_cupy_CHROM = pd.DataFrame(data_cupy_CHROM)
    frame_extraction = pd.DataFrame(data_gt, index = data_gt['time gt'])
    return (frame_cupy_POS, frame_torch_POS, frame_cupy_CHROM, frame_extraction)


dir = './metrics_datasets/'

#video = 0

for video in range(0,len(allvideo)):

    person = titles[video][0]
    activity = titles[video][1]
    print("Procesing video n°"+str(video)+" of "+str(range(21,len(allvideo))[1]))
    print("Generating data about "+person+' '+activity+'...') 

    #Generate data from prediction and GT
    data_cupy_CHROM, data_torch_CHROM, data_cupy_POS, data_gt = get_video_data('convexhull','holistic', video_index= video)

    #Create DataFrame from previous data
    frame_cupy_CHROM, frame_torch_CHROM, frame_cupy_POS, frame_gt = get_video_dataframes(data_cupy_CHROM, data_torch_CHROM, data_cupy_POS, data_gt)

    #Create CSV for each DataFrame
    if not os.path.exists(dir):
        os.mkdir(dir)

    rPPG_method = f'{frame_cupy_CHROM=}'.split('=')[0].split('_')
    frame_cupy_CHROM.to_csv(dir+person+'_'+activity+'('+ rPPG_method[1]+rPPG_method[2] +')'+'.csv', index= False)

    rPPG_method = f'{frame_torch_CHROM=}'.split('=')[0].split('_')
    frame_torch_CHROM.to_csv(dir+person+'_'+activity+'('+ rPPG_method[1]+rPPG_method[2] +')'+'.csv', index= False)

    rPPG_method = f'{frame_cupy_POS=}'.split('=')[0].split('_')
    frame_cupy_POS.to_csv(dir+person+'_'+activity+'('+ rPPG_method[1]+rPPG_method[2] +')'+'.csv', index= False)

    frame_gt.to_csv(dir+person+'_'+activity+'(extraction_GT)'+'.csv', index= False)