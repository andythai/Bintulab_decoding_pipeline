
#activate cellpose&&python "C:\Scripts\NMERFISH\worker_Scope3__XXBB.py"

from multiprocessing import Pool, TimeoutError
import time, sys
import os, sys, numpy as np

master_analysis_folder = r'C:\\Scripts\\NMERFISH' ## where the code libes
lib_fl = r'C:\\Scripts\\NMERFISH\\codebooks\\custom_rig_300GP_21_bits_new.csv' ### codebook
### Did you compute PSF and median flat field images?
#psf_file = r'C:\Scripts\NMERFISH\psfs\psf_750_Scope3_final.npy' # PSF based  on data form Scope at Bintu lab
psf_file = r'C:\\Scripts\\Scope_Zik1_params\\psf\\psf_cy3_zik1_shortTube.npy'

#master_data_folders = [r'Y:\Lab\MERFISH_Imaging_data\06_18_2023_RzA3A6_1_hemi' ]
master_data_folders = [r'Y:\\Lab\\MERFISH_Imaging_data\\20240406_MD_191']
#save_folder =r'\\192.168.0.7\bbfishmahsa3\CGBB_embryo_4_28_2023\MERFISH_AnalysisP12'
save_folder = os.path.join(master_data_folders[-1],'MERFISH_Analysis')
iHm=1 #H iHmin -> H iHmax oly keeps folders of the the form H33,H34...
iHM=7 

sys.path.append(master_analysis_folder)
from ioMicro import *

def compute_drift(save_folder,fov,all_flds,set_,redo=False,gpu=False):
    """
    save_folder where to save analyzed data
    fov - i.e. Conv_zscan_005.zarr
    all_flds - folders that contain eithger the MERFISH bits or control bits or smFISH
    set_ - an extra tag typically at the end of the folder to separate out different folders
    """
    #print(len(all_flds))
    #print(all_flds)
    
    # defulat name of the drift file 
    drift_fl = save_folder + os.sep + 'driftNew_' + fov.split('.')[0] + ' --' + set_ + '.pkl'
    
    iiref = None
    fl_ref = None
    previous_drift = {}
    if not os.path.exists(drift_fl) or redo:
        redo = True
    else:
        try:
            drifts_, all_flds_, fov_, fl_ref = pickle.load(open(drift_fl,'rb'))
            all_tags_ = np.array([os.path.basename(fld) for fld in all_flds_])
            all_tags = np.array([os.path.basename(fld) for fld in all_flds])
            iiref = np.argmin([np.sum(np.abs(drift[0])) for drift in drifts_])
            previous_drift = {tag:drift for drift, tag in zip(drifts_, all_tags_)}

            if not (len(all_tags_)==len(all_tags)):
                redo = True
            else:
                if not np.all(np.sort(all_tags_)==np.sort(all_tags)):
                    redo = True
        except:
            os.remove(drift_fl)
            redo=True
    if redo:
        previous_drift={}
        fls = [fld+os.sep+fov for fld in all_flds]
        if fl_ref is None:
            #fl_ref = fls[len(fls)//2]  # this assigns the middle hybridization round as the reference round. Sometime better to instead use hyb1 as reference
            fl_ref = fls[0]
        obj = None
        newdrifts = []
        all_fldsT = []
        print("Files to perform drift correction:",fls)
        for fl in tqdm(fls):
            fld = os.path.dirname(fl)
            tag = os.path.basename(fld)
            new_drift_info = previous_drift.get(tag,None)
            if new_drift_info is None:
                if obj is None:
                    obj = fine_drift(fl_ref,fl,sz_block=600)
                else:
                    obj.get_drift(fl_ref,fl)
                new_drift = -(obj.drft_minus+obj.drft_plus)/2
                new_drift_info = [new_drift,obj.drft_minus,obj.drft_plus,obj.drift,obj.pair_minus,obj.pair_plus]
            #print(new_drift_info)
            newdrifts.append(new_drift_info)
            all_fldsT.append(fld)
            pickle.dump([newdrifts,all_fldsT,fov,fl_ref],open(drift_fl,'wb'))
def compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    fls = [fld+os.sep+fov for fld in all_flds]
    for fl in fls:
        icol=3
        #im_med_fl = save_folder+os.sep+'med_col_raw'+str(icol)+'.npz'#### mistake
        im_med_fl = None #  no med_fl for single FOV experiment
        get_dapi_features(fl,save_folder,set_,gpu=gpu,im_med_fl = im_med_fl,
                    psf_fl = psf_file)
                    


def get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5):
    
    obj = get_dapi_features(fl,save_folder,set_)
    obj_ref = get_dapi_features(fl_ref,save_folder,set_)
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0,0
    if (len(obj.Xh_plus)>0) and (len(obj_ref.Xh_plus)>0):
        X = obj.Xh_plus[:,:3]
        X_ref = obj_ref.Xh_plus[:,:3]
        tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if (len(obj.Xh_minus)>0) and (len(obj_ref.Xh_minus)>0):
        X = obj.Xh_minus[:,:3]
        X_ref = obj_ref.Xh_minus[:,:3]
        tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if np.max(np.abs(tzxy_minus-tzxy_plus))<=2:
        tzxyf = -(tzxy_plus*N_plus+tzxy_minus*N_minus)/(N_plus+N_minus)
    else:
        tzxyf = -[tzxy_plus,tzxy_minus][np.argmax([N_plus,N_minus])]
    

    return [tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus]

def compute_drift_V2(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    drift_fl = save_folder+os.sep+'driftNew_'+fov.split('.')[0]+'--'+set_+'.pkl'
    if not os.path.exists(drift_fl) or redo:
        fls = [fld+os.sep+fov for fld in all_flds]
        #fl_ref = fls[len(fls)//2]  # this assigns the middle hybridization round as the reference round.
                                   # But sometime better to instead use hyb1 as reference
        fl_ref = fls[0]                           
        newdrifts = []
        for fl in fls:
            drft = get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5)
            print(drft)
            newdrifts.append(drft)
        pickle.dump([newdrifts,all_flds,fov,fl_ref],open(drift_fl,'wb'))
def main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method):
    im_ = read_im(fld+os.sep+fov)
    im__ = np.array(im_[icol],dtype=np.float32)
    
    if old_method:
        ### previous method
        im_n = norm_slice(im__,s=30)
        #Xh = get_local_max(im_n,500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,dbscan=True,
        #      return_centers=False,mins=None,sigmaZ=1,sigmaXY=1.5)
        Xh = get_local_maxfast_tensor(im_n,th_fit=500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5,gpu=False)
    else:
        ### new method
        fl_med = save_folder+os.sep+'med_col_raw'+str(icol)+'.npz'#### mistake
        if os.path.exists(fl_med):
            im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
            im_med = cv2.blur(im_med,(20,20))
            im__ = im__/im_med*np.median(im_med)
        try:
            Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=True,
                                    deconv={'method':'wiener','beta':0.0001},
                                    delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
            
        except:
            Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=False,
                                    deconv={'method':'wiener','beta':0.0001},
                                    delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
    np.savez_compressed(save_fl,Xh=Xh)
def compute_fits(save_folder,fov,all_flds,redo=False,ncols=4,
                psf_file = psf_file,try_mode=True,old_method=False):
    psf = np.load(psf_file)
    
    for fld in tqdm(all_flds):
        for icol in range(ncols-1):
            tag = os.path.basename(fld)
            save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npz'
            if not os.path.exists(save_fl) or redo:
                if try_mode:
                    try:
                        main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method)
                    except:
                        print("Failed",fld,fov,icol)
                else:
                    main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method)
                        
def compute_decoding(save_folder,fov,set_,redo=False):
    dec = decoder_simple(save_folder,fov,set_)
    #self.decoded_fl = self.decoded_fl.replace('decoded_','decodedNew_')
    complete = dec.check_is_complete()
    if complete==0 or redo:
        #compute_drift(save_folder,fov,all_flds,set_,redo=False,gpu=False)
        dec = decoder_simple(save_folder,fov=fov,set_=set_)
        dec.get_XH(fov,set_,ncols=3,nbits=7,th_h=0)#number of colors match ######################################################## Could change for speed.
        dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
        dec.load_library(lib_fl,nblanks=-1)
        
        dec.ncols = 3
        if False:
            dec.XH_save = dec.XH.copy()
            keep_best_N_for_each_Readout(dec,Nkeep = 15000)
            dec.get_inters(dinstance_th=5,enforce_color=True)# enforce_color=False
            dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=-1)
            apply_fine_drift(dec,plt_val=True)
            dec.XH = dec.XH_save.copy()
            R = dec.XH[:,-1].astype(int)
            dec.XH[:,:3] -= dec.drift_arr[R]
        #dec.get_inters(dinstance_th=2,enforce_color=True)# enforce_color=False
        dec.get_inters(dinstance_th=2,nmin_bits=4,enforce_color=False,redo=True)
        #dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=None,nbits=24)#,is_unique=False)
        get_icodesV2(dec,nmin_bits=4,delta_bits=None,iH=-3,redo=False,norm_brightness=-2,nbits=21,is_unique=True)

def get_iH(fld): 
    try:
        return int(os.path.basename(fld).split('_')[0][1:])
    except:
        return np.inf

def get_files(set_ifov,iHm=iHm,iHM=iHM):
    
    
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    
    #master_folder = master_data_folder
    all_flds = []
    for master_folder in master_data_folders:
        all_flds += glob.glob(master_folder+r'\H*MER*')
    #all_flds += glob.glob(master_folder+r'\P*')
    
    ### reorder based on hybe
    all_flds = np.array(all_flds)[np.argsort([get_iH(fld)for fld in all_flds])] 
    
    set_,ifov = set_ifov
    all_flds = [fld for fld in all_flds if set_ in os.path.basename(fld)]
    
    all_flds = [fld for fld in all_flds if ((get_iH(fld)>=iHm) and (get_iH(fld)<=iHM))]
    
    fovs_fl = save_folder+os.sep+'fovs__'+set_+'.npy'
    if not os.path.exists(fovs_fl):
        folder_map_fovs = all_flds[0]#[fld for fld in all_flds if 'low' not in os.path.basename(fld)][0]
        fls = glob.glob(folder_map_fovs+os.sep+'*.zarr')
        fovs = np.sort([os.path.basename(fl) for fl in fls])
        np.save(fovs_fl,fovs)
    else:
        fovs = np.sort(np.load(fovs_fl))
    fov=None
    if ifov<len(fovs):
        fov = fovs[ifov]
        all_flds = [fld for fld in all_flds if os.path.exists(fld+os.sep+fov)]
    return save_folder,all_flds,fov
        

def compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method):
    print("Computing fitting on: "+str(fov))
    print(len(all_flds),all_flds)
    compute_fits(save_folder,fov,all_flds,redo=redo_fits,try_mode=try_mode,old_method=old_method)
    print("Computing drift on: "+str(fov))
    
    # line below is permanetly commented
    #compute_drift(save_folder,fov,all_flds,set_,redo=redo_drift)
    
    
    compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=False)
    compute_drift_V2(save_folder,fov,all_flds,set_,redo=redo_drift,gpu=False)
    compute_decoding(save_folder,fov,set_,redo=redo_decoding)

def main_f(set_ifov,redo_fits = False,redo_drift=False,redo_decoding=True,try_mode=True,old_method=False):
    set_,ifov = set_ifov
    save_folder,all_flds,fov = get_files(set_ifov)
    
    if try_mode:
        try:
            compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method)
        except:
            print("Failed within the main analysis:")
    else:
        compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method)
    
    return set_ifov
    
    
    
if __name__ == '__main__':
    # start 4 worker processes
   
    run_fovs = np.arange(700)
    

    #run_fovs = np.load(os.path.join(save_folder,'pre_processing','tdTomato_stained.npy'),allow_pickle = True)
    
    items = [('',ifov) for ifov in run_fovs] # 1500 FOVS
    
    
    #main_f(['',614],redo_fits = True,redo_drift=True,redo_decoding=True,try_mode=False)
    
    if True:
       with Pool(processes=4) as pool:
           print('starting pool')
           result = pool.map(main_f, items)