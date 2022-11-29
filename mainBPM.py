import ast
import numpy as np
from pyVHR.extraction.sig_processing import *
from pyVHR.extraction.sig_extraction_methods import *
from pyVHR.extraction.skin_extraction_methods import *
from pyVHR.BVP.BVP import *
from pyVHR.BPM.BPM import *
from pyVHR.BVP.methods import *
from pyVHR.BVP.filters import *


from pyVHR.realtime.params import Params
import time

from inspect import getmembers, isfunction

roi_method='convexhull'
roi_approach='holistic' 
method='cupy_POS'
bpm_type='welch' 
pre_filt=False
post_filt=True
fps = 25
bpm_plot = []
bpm_save = []
bpm_plot_max = 60
image = None
original_image = None
skin_image = None
patches_image = None
vhr_t = None


type_options = ['mean', 'median']
patches_options = ['squares','rects']
rects_dim_sim = "[[],]"
skin_color_low_threshold = "75"
skin_color_high_threshold = "230"
sig_color_low_threshold = "75"
sig_color_high_threshold = "230"

visualizeskintrue_sim = False
visualizeldmkstrue_sim = False
visualizepatchestrue_sim = False
visualizeldmksnumtrue_sim = False
fontSize = "0.3"  ## to correct: allways False when isnumeric()
fontcolor = '(255, 0, 0, 255)'
color_low_threshold = "75"
color_high_threshold = "230"


bpm_plot_max = 10
   
def apply_params():
    
    Params.fake_delay = True
    Params.cuda = cuda
    Params.fps_fixed = fps
    Params.rPPG_method = method
    Params.skin_extractor = roi_method
    Params.approach = roi_approach
    Params.type = type_options[0]
    Params.patches = patches_options[0]
    
    try:
        Params.rects_dims = ast.literal_eval(rects_dim_sim)
        # Default if len is not the same
        if len(Params.rects_dims) != len(Params.landmarks_list):
            new_rect_dim = []
            for i in range(len(Params.landmarks_list)):
                new_rect_dim.append(
                    [Params.squares_dim, Params.squares_dim])
            Params.rects_dims = new_rect_dim
    except:
        # Default if parameter is wrong
        new_rect_dim = []
        for i in range(len(Params.landmarks_list)):
            new_rect_dim.append(
                [Params.squares_dim, Params.squares_dim])
        Params.rects_dims = new_rect_dim
    if skin_color_low_threshold.isnumeric():
        Params.skin_color_low_threshold = int(
            skin_color_low_threshold) if int(skin_color_low_threshold) >= 0 and int(skin_color_low_threshold) <= 255 else 2
    if skin_color_high_threshold.isnumeric():
        Params.skin_color_high_threshold = int(
            skin_color_high_threshold) if int(skin_color_high_threshold) >= 0 and int(skin_color_high_threshold) <= 255 else 254
    if sig_color_low_threshold.isnumeric():
        Params.sig_color_low_threshold = int(
            sig_color_low_threshold) if int(sig_color_low_threshold) >= 0 and int(sig_color_low_threshold) <= 255 else 2
    if sig_color_high_threshold.isnumeric():
        Params.sig_color_high_threshold = int(
            sig_color_high_threshold) if int(sig_color_high_threshold) >= 0 and int(sig_color_high_threshold) <= 255 else 254
    if color_low_threshold.isnumeric():
        Params.color_low_threshold = int(
            color_low_threshold) if int(color_low_threshold) >= 0 and int(color_low_threshold) <= 255 else -1
    if color_high_threshold.isnumeric():
        Params.color_high_threshold = int(
            color_high_threshold) if int(color_high_threshold) >= 0 and int(color_high_threshold) <= 255 else 255
    
    
    Params.visualize_skin = True if bool(
        visualizeskintrue_sim) else False
    Params.visualize_landmarks = True if bool(
        visualizeldmkstrue_sim) else False
    Params.visualize_patches = True if bool(
        visualizepatchestrue_sim) else False
    Params.visualize_landmarks_number = True if bool(
        visualizeldmksnumtrue_sim) else False
    try:
        colorfont = ast.literal_eval(fontcolor)
        if len(colorfont) == 4:
            correct = True
            for e in colorfont:
                if not(e >= 0 and e <= 255):
                    correct = False
            if correct:
                Params.font_color = colorfont
    except:
        pass
    if fontSize.isnumeric():
        Params.font_size = float(
            fontSize) if float(fontSize) > 0.0 else 0.3
    

def start_predict():
    bpm_plot = []
    bpm_save = []
    image = None

import queue
from pyVHR.extraction.sig_processing import *
from pyVHR.extraction.sig_extraction_methods import *
from pyVHR.extraction.skin_extraction_methods import *

def analisys(frame_input):
    q_bpm = queue.Queue()
    #q_video_image = queue.Queue()
    #q_skin_image = queue.Queue()
    #q_patches_image = queue.Queue()
    q_stop = queue.Queue()
    #q_stop_cap = queue.Queue()
    q_frames = queue.Queue()

    #bpm_list = []

    sig_ext_met = None
    ldmks_regions = None
    # Holistic settings #
    if Params.approach == 'holistic':
        sig_ext_met = holistic_mean
    # Patches settings #
    elif Params.approach == 'patches':
        # extraction method
        if Params.type == "mean" and Params.patches == "squares":
            sig_ext_met = landmarks_mean
        elif Params.type == "mean" and Params.patches == "rects":
            sig_ext_met = landmarks_mean_custom_rect
        elif Params.type == "median" and Params.patches == "squares":
            sig_ext_met = landmarks_median
        elif Params.type == "median" and Params.patches == "rects":
            sig_ext_met = landmarks_median_custom_rect
        # patches dims
        if Params.patches == "squares":
            ldmks_regions = np.float32(Params.squares_dim)
        elif Params.patches == "rects":
            ldmks_regions = np.float32(Params.rects_dims)

    SignalProcessingParams.RGB_LOW_TH = np.int32(
        Params.sig_color_low_threshold)
    SignalProcessingParams.RGB_HIGH_TH = np.int32(
        Params.sig_color_high_threshold)
    SkinProcessingParams.RGB_LOW_TH = np.int32(
        Params.skin_color_low_threshold)
    SkinProcessingParams.RGB_HIGH_TH = np.int32(
        Params.skin_color_high_threshold)

    color = np.array([Params.font_color[0],
                      Params.font_color[1], Params.font_color[2]], dtype=np.uint8)

    skin_ex = None
    target_device = 'GPU' if Params.cuda else 'CPU'
    if Params.skin_extractor == 'convexhull':
        skin_ex = SkinExtractionConvexHull(target_device)
    elif Params.skin_extractor == 'faceparsing':
        skin_ex = SkinExtractionFaceParsing(target_device)

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    PRESENCE_THRESHOLD = 0.5
    VISIBILITY_THRESHOLD = 0.5
    
    fps = Params.fps_fixed
    tot_frames = int(Params.tot_sec*fps)
    #print("tot frames: ",tot_frames)

    sig = []
    processed_frames_count = 0
    sig_buff_dim = int(fps * Params.winSize)
    sig_stride = int(fps * Params.stride)
    sig_buff_counter = sig_stride # stride en un segundo. Relacionado a FPS

    BPM_obj = None

    timeCount = []

    send_images_count = 0
    send_images_stride = 3

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        counter = 0
        while True:
            counter +=1
            #print("-----  Iteration counter: ", counter)
            start_time = time.perf_counter()*1000
            if q_frames.empty():
                for f in range(len(frame_input)):
                    q_frames.put(frame_input[f])

            frame = None

            if not q_frames.empty():  
                frame = q_frames.get()
                #frame = q_frames.queue[0]
                if type(frame) == int:  
                    return(-3)
            if not q_stop.empty(): 
                q_stop.get()
                return(-2)
            if frame is None:
                #print("Frame is None.")
                return(-1)

            # convert the BGR image to RGB.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frames_count += 1
            #print("processed_frames_count: ", processed_frames_count)
            width = image.shape[1]
            height = image.shape[0]
            # [landmarks, info], with info->x_center ,y_center, r, g, b
            ldmks = np.zeros((468, 5), dtype=np.float32)
            ldmks[:, 0] = -1.0
            ldmks[:, 1] = -1.0
            magic_ldmks = []
            ### face landmarks ###
            results = face_mesh.process(image)
            if results.multi_face_landmarks:
                #print("Hay resultados multi face landmarks")
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [l for l in face_landmarks.landmark]
                for idx in range(len(landmarks)):
                    landmark = landmarks[idx]
                    if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                            or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                        coords = mp_drawing._normalized_to_pixel_coordinates(
                            landmark.x, landmark.y, width, height)
                        if coords:
                            ldmks[idx, 0] = coords[1]
                            ldmks[idx, 1] = coords[0]
                ### skin extraction ###
                cropped_skin_im, full_skin_im = skin_ex.extract_skin(
                    image, ldmks)
            else:
                #print("ATENCIÓN: no hay resultados multi face landmarks")
                cropped_skin_im = np.zeros_like(image)
                full_skin_im = np.zeros_like(image)
            ### SIG ###
            if Params.approach == 'patches':
                magic_ldmks = np.array(
                    ldmks[Params.landmarks_list], dtype=np.float32)
                temp = sig_ext_met(magic_ldmks, full_skin_im, ldmks_regions,
                                np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH))
                temp = temp[:, 2:]  # keep only rgb mean
            elif Params.approach == 'holistic':
                #print("enter holistic")
                temp = sig_ext_met(cropped_skin_im, np.int32(
                    SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH))
            sig.append(temp)
            #print("sig buff dim: ", sig_buff_dim)
            if processed_frames_count > sig_buff_dim:
                #print("now processed_frames_count > sig_buff_dim:")
                sig = sig[1:]
                #print("sig_buff_counter: ",sig_buff_counter)
                if sig_buff_counter == 0:
                    sig_buff_counter = sig_stride
                    copy_sig = np.array(sig, dtype=np.float32)
                    copy_sig = np.swapaxes(copy_sig, 0, 1)
                    copy_sig = np.swapaxes(copy_sig, 1, 2)
                    ### Pre_filtering ###
                    if Params.approach == 'patches':
                        copy_sig = rgb_filter_th(copy_sig, **{'RGB_LOW_TH':  np.int32(Params.color_low_threshold),
                                                            'RGB_HIGH_TH': np.int32(Params.color_high_threshold)})
                    for filt in Params.pre_filter:
                        #print("pre filt: ", filt)
                        if filt != {}:
                            if 'fps' in filt['params'] and filt['params']['fps'] == 'adaptive' and fps is not None:
                                filt['params']['fps'] = float(fps)
                            if filt['params'] == {}:
                                copy_sig = filt['filter_func'](
                                    copy_sig)
                            else:
                                copy_sig = filt['filter_func'](
                                    copy_sig, **filt['params'])
                    ### BVP ###
                    bvp = np.zeros((0, 1), dtype=np.float32)
                    if Params.method['device_type'] == 'cpu':
                        bvp = signals_to_bvps_cpu(
                            copy_sig, Params.method['method_func'], Params.method['params'])
                    elif Params.method['device_type'] == 'torch':
                        bvp = signals_to_bvps_torch(
                            copy_sig, Params.method['method_func'], Params.method['params'])
                    elif Params.method['device_type'] == 'cuda':
                        bvp = signals_to_bvps_cuda(
                            copy_sig, Params.method['method_func'], Params.method['params'])
                    ### Post_filtering ###
                    for filt in Params.pre_filter:
                        #print("post filt: ", filt)
                        if filt != {}:
                            bvp = np.expand_dims(bvp, axis=1)
                            if 'fps' in filt['params'] and filt['params']['fps'] == 'adaptive' and fps is not None:
                                filt['params']['fps'] = float(fps)
                            if filt['params'] == {}:
                                bvp = filt['filter_func'](bvp)
                            else:
                                bvp = filt['filter_func'](
                                    bvp, **filt['params'])
                            bvp = np.squeeze(bvp, axis=1)
                    ### BPM ###
                    if Params.cuda:
                        bvp_device = cupy.asarray(bvp)
                        if BPM_obj == None:
                            BPM_obj = BPMcuda(bvp_device, fps,
                                            minHz=Params.minHz, maxHz=Params.maxHz)
                        else:
                            BPM_obj.data = bvp_device
                        if Params.BPM_extraction_type == "welch":
                            bpm = BPM_obj.BVP_to_BPM()
                            bpm = cupy.asnumpy(bpm)
                        elif Params.BPM_extraction_type == "psd_clustering":
                            bpm = BPM_obj.BVP_to_BPM_PSD_clustering()
                    else:
                        if BPM_obj == None:
                            BPM_obj = BPM(bvp, fps, minHz=Params.minHz,
                                        maxHz=Params.maxHz)
                        else:
                            BPM_obj.data = bvp
                        if Params.BPM_extraction_type == "welch":
                            bpm = BPM_obj.BVP_to_BPM()
                        elif Params.BPM_extraction_type == "psd_clustering":
                            bpm = BPM_obj.BVP_to_BPM_PSD_clustering()
                    if Params.approach == 'patches':  # Median of multi BPMs
                        if len(bpm.shape) > 0 and bpm.shape[0] == 0:
                            bpm = np.float32(0.0)
                        else:
                            bpm = np.float32(np.median(bpm))
                    q_bpm.put(bpm)
                    #print("processed frames: ",processed_frames_count)
                    print("bpm: ",bpm)
                    #bpm_list.append(bpm)
                    return bpm

                else: # si signal buff counter no es 0:
                    sig_buff_counter -= 1

            end_time = time.perf_counter()*1000
            timeCount.append(end_time-start_time)
            if len(timeCount) > 100:
                timeCount = timeCount[1:]
            ### loop break ###
            if tot_frames is not None and tot_frames > 0 and processed_frames_count >= tot_frames:
                #print("tot frames not None and tot frames > 0, processed_frames_count >= tot frames")
                #return bpm_list,q_bpm
                pass

