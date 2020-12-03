import pandas
import numpy as np
import torch
from kmeans_pytorch import kmeans
import os
import cv2
import glob

# topcon_candidates = pandas.read_csv('/home/ubuntu/data/Workspace/Soobin/data/resolution/topcon_candidate_higher_res.csv')

# total_cnt = len(topcon_candidates['file_path'])
# length = int(total_cnt/20)
# img_path = '/home/ubuntu/data/OCT/Seoul/'

def create_tempfeatures(idx):
    # temp = np.zeros((length,256*256*3+1),dtype='float32')
    if idx == 19:
        st_idx = length*idx
        end_idx = total_cnt
    else:
        st_idx = length*idx
        end_idx = length*(idx+1)
    print(st_idx,"  ",end_idx)
    transformation = np.zeros((end_idx-st_idx,20000*3),dtype='float32')

    for i, idx in enumerate (list(range(st_idx,end_idx))):
        path = topcon_candidates['file_path'][idx]
        path = '/'.join(path.split('/')[4:])
        im = cv2.imread(img_path + path)
        im = np.array(cv2.resize(im,dsize=(200,100)),dtype='float32')
        # h, edges = np.histogramdd(im.reshape(-1,3),8,normed=True,range=[(0,255),(0,255),(0,255)])
        transform=np.fft.fft2(im)
        transformation[i] = np.abs(transform.flatten())
    
    res = torch.from_numpy(transformation)  # to use gpu
    return res

csv = []
def topcon_imwrite(cluster, idxx, idx):
    if idxx == 19:
        st_idx = length*idxx
        end_idx = total_cnt
    else:
        st_idx = length*idxx
        end_idx = length*(idxx+1)
    print(st_idx,"  ",end_idx)

    path = topcon_candidates['file_path'][st_idx+idx]
    path = '/'.join(path.split('/')[4:])
    path = img_path+path
    im = cv2.imread(path)

    name = topcon_candidates['file_name'][st_idx+idx]

    item = []
    item.append(path)
    item.append(name)
    item.append(str(cluster))
    csv.append(item)
    
    savepath ='./topcon_candidates_5class_higher/attempt'+str(idxx)+'/img'
    savepath = savepath+str(cluster)+'/'
    
    
    os.makedirs(savepath,exist_ok=True)
    cv2.imwrite(savepath+name,im)


def delete_existing_img(path):
    dirs = glob.glob(path+'*')
    directory = [dire for dire in dirs if not '.jpg' in dire]
    directory = [dire for dire in directory if not '.csv' in dire]
    print(directory)
    image = [img for img in dirs if 'jpg' in img]

    print(len(image))
    for dire in directory:
        directory_img = glob.glob(dire+'/*/*')
        for img in directory_img:
            search_img = img.split('/')[10]
            for imgg in image:
                if search_img in imgg:
                    os.remove(imgg)
                    image.remove(imgg)
    print(len(image))
    return


def delete_existing_folder():
    del_attempt_img = [(0,0),(9,4),(12,2),(14,4),(18,3),(19,3)]
    import shutil
    for (attempt, img) in del_attempt_img:
        del_path = '/home/ubuntu/data/Workspace/Soobin/data/topcon_wide_higher_resolution/attempt'+str(attempt)+'/img'+str(img)
        print(del_path)
        shutil.rmtree(del_path)


def coexist_multiclass(to_class):
    path='/home/ubuntu/data/Workspace/Soobin/wide_total/'
    path = glob.glob(path+'*')
    tot_cnt = len(path)
    print(tot_cnt)
    length = int(tot_cnt/50)
    for attempt in range(24,50):
        csv=[]
        if attempt == 49:
            st_idx = length*attempt
            end_idx = tot_cnt
        else:
            st_idx = length*attempt
            end_idx = length*(attempt+1)
        print(st_idx,"  ",end_idx)
        transformation = np.zeros((end_idx-st_idx,20000*3),dtype='float32')

        for i, idx in enumerate (list(range(st_idx,end_idx))):
            im = cv2.imread(path[idx])
            im = np.array(cv2.resize(im,dsize=(200,100)),dtype='float32')
            # h, edges = np.histogramdd(im.reshape(-1,3),8,normed=True,range=[(0,255),(0,255),(0,255)])
            transform=np.fft.fft2(im)
            transformation[i] = np.abs(transform.flatten())
        res = torch.from_numpy(transformation)
        print("start clustering")
        labels,_ = kmeans(X=res, num_clusters=to_class, distance='euclidean', device=torch.device('cuda:1'))
        cluster_map = pandas.DataFrame()
        cluster_map['cluster'] = labels

        for i, idx in enumerate (list(range(st_idx,end_idx))):
            im_path = path[idx]
            im = cv2.imread(im_path)
            name = path[idx].split('/')[7]
            cluster = cluster_map['cluster'][i]
            savepath ='/home/ubuntu/data/Workspace/Soobin/attempt'+str(attempt)+'/img'
            savepath = savepath+str(cluster)+'/'
            os.makedirs(savepath,exist_ok=True)
            cv2.imwrite(savepath+name,im)
            # item = []
            # item.append(path)
            # item.append(name)
            # item.append(str(cluster))
            # csv.append(item)
        # csv_file = pandas.DataFrame(csv, columns=['file_path', 'file_name','cluster'])
        # csv_file.to_csv('/home/ubuntu/data/Workspace/Soobin/attempt%d/'%(attempt)+'cluster.csv', index=False, encoding='cp949')
    return


coexist_multiclass(6)
# path='/home/ubuntu/data/Workspace/Soobin/wide_total/'
# path = glob.glob(path+'*')
# # print(path)
# for idx, p in enumerate(path):
#     name = path[idx].split('/')[8]
#     print(name)
#     # if 'wide_total' in name:
#     #     split = path[idx].split('/')[:7]
#     #     rename_path = '/'.join(split)
#     #     rename = name[14:]
#     #     print(rename)
#     #     os.rename(path[idx],rename_path+'/'+rename)

def to_csv(path):
    for f in path:
        item = []
        name = f.split('/')[10][:-4]
        parms = name.split('_')
        code = parms[0]
        year = parms[2][:4]
        month= parms[2][4:6]
        date = parms[2][6:]
        item.append('OCT')
        item.append('Seoul')
        item.append(year)
        item.append(month)
        item.append(date)
        item.append(code)
        item.append(name)
        csv.append(item)
        print(item)
    csv_file = pandas.DataFrame(csv, columns=['root', 'hospital','year', 'month', 'date', 'code', 'file'])
    csv_file.to_csv(savepath+'wide_higher_resolution.csv', index=False, encoding='cp949')
    return



def move_result():
    path = '/home/ubuntu/data/Workspace/Soobin/data/topcon_wide_higher_resolution/'
    files = glob.glob(path+'*/*/*')
    print(files[:30])
    savepath ='/home/ubuntu/data/Workspace/Soobin/wide_higher_resolution/data/'
    for f in files:
        im = cv2.imread(f)
        name = f.split('/')[10][:-4]
        cv2.imwrite(savepath+name,im)
        # os.remove(f)
    return


    # length = len(path)
    # transformation = np.zeros((len(path),20000*3),dtype='float32')
    
    # for p in path:
    #     im = cv2.imread(p)
    #     im = np.array(cv2.resize(im,dsize=(200,100)),dtype='float32')
    #     # h, edges = np.histogramdd(im.reshape(-1,3),8,normed=True,range=[(0,255),(0,255),(0,255)])
    #     transform=np.fft.fft2(im)
    #     transformation[i] = np.abs(transform.flatten())
    
    # res = torch.from_numpy(transformation)  # to use gpu
    # print("start clustering")
    # labels,_ = kmeans(X=topcon, num_clusters=to_class, distance='euclidean', device=torch.device('cuda:0'))
    # cluster_map = pandas.DataFrame()
    # cluster_map['cluster'] = labels

    # for i in range (0,len(cluster_map['cluster'])):
    #         topcon_imwrite(cluster_map['cluster'][i],j, i)


def tot_dat():
    for j in range(0,20):
        topcon = create_tempfeatures(j)
        print("start clustering")
        labels,_ = kmeans(X=topcon, num_clusters=5, distance='euclidean', device=torch.device('cuda:0'))
        cluster_map = pandas.DataFrame()
        cluster_map['cluster'] = labels
        for i in range (0,len(cluster_map['cluster'])):
            topcon_imwrite(cluster_map['cluster'][i],j, i)
        csv_file = pandas.DataFrame(csv, columns=['file_path', 'file_name','cluster'])
        csv=[]
        csv_file.to_csv('./topcon_candidates_5class_higher/attempt%d/'%(j)+'cluster.csv', index=False, encoding='cp949')
        print("one attempt done!")
