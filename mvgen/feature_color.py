from msilib.schema import Error
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os, glob, sys
import json
from scipy.stats import itemfreq
from sklearn.cluster import KMeans

from .config import (CLUSTERS, STATISTICS_PATH, 
                    THUMBNAILS_PATH, MIN_VIDEOS_IN_STYLE)

'''
EXTRACT_FEATURE : for one video file, compute the mean color histogram
path (string) : path of video file
saveThumbnail (bool) : save or not an image representing the video (useful for representing kmeans result)
'''
def extract_feature(path, saveThumbnail): 

    # We will use image every sampling frame to compute the mean histogram
    sampling = 5

    # Path to video file
    if os.path.exists(path):
        vidObj = cv2.VideoCapture(path) #打开path的视频
        numFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))#获取该视频的总帧数
        numSec = numFrames/vidObj.get(cv2.CAP_PROP_FPS) #视频秒数=总帧数/帧速率（每秒多少帧） 

        # Do not use scenes too short
        if numFrames<12:
            print("Scene too short (<25 frames, ~500ms). No analyzing")
            return (numSec, np.empty((0,0))) # （视频秒数，[]）

        # Do not use scenes too long
        elif numFrames>125:
            print("Scene too long (>125 frames, ~5 sec). No analyzing")
            return (numSec, np.empty((0,0))) # (视频秒数，[])

        else:
            print("Analyzing video %s : %d frames, %f seconds ..."%(os.path.basename(path),numFrames,numSec)) #path最后的文件名，帧数，秒数

            # Used as counter variable 
            count = 0 
            deviate = 3 # Not start at 0 because of transition frames

            # checks whether frames were extracted 
            success = 1

            # We stack all images on imgBase.
            imgBase = np.zeros((150,150,3), np.uint8) 

            while success: 
                success, image = vidObj.read()  #success表示成功读取，image表示按帧读取的帧

                # When we encounter a frame we want to analyse : stack image
                if count%sampling==deviate and count<numFrames:    
                    numImg = (count-deviate)//sampling #第numImg个要分析的帧
                    # print("New image: %d"%numImg)

                    if numImg==0:
                        imgBase = cv2.resize(image, (0,0), fx=0.2, fy=0.2) #将图像沿横纵坐标各缩小五倍
                    
                    else:
                        img = cv2.resize(image, (0,0), fx=0.2, fy=0.2) #将图像沿横纵坐标各缩小五倍
                        imgBase = np.concatenate((imgBase, img), axis=1) #将imgBase和img拼接

                count += 1 # Keeps track of number of frames

                if saveThumbnail and count==numFrames//2 :
                    cv2.imwrite(THUMBNAILS_PATH+os.path.basename(path)+".jpg",image)
            
            # imgBase = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB)
            hist = compute_histogram(imgBase)

            if hist[0] == 1 and hist[256] == 1 and hist[512] == 1:   #纯黑的帧不会采纳
                return (numSec, np.empty((0,0)))     #（视频秒数，[]）
            else:
                return (numSec, hist)   #(视频秒数，颜色分布直方)


'''
Return an numpy array size 256 x 3 which is the color histogram of the picture :
[0,255] = blue
[256,511] = green
[512,767] = red
for each channel [0,255], index i gives the percentage of pixels with intensity i in the pic
'''    #输入一个长图，计算其颜色分布，用以代表子视频的feature，之后进行选择的时候依据此来进行选择
def compute_histogram(img):
    color = ('b','g','r')
    hist = np.empty((0,256))
    # fig = plt.figure()
    for i,col in enumerate(color):
        histcolor = cv2.calcHist([img],[i],None,[256],[0,256])
        hist = np.append(hist, histcolor/np.linalg.norm(histcolor))
        # plt.plot(histcolor,color = col)
        # plt.xlim([0,256])
    # plt.show()
    # plt.close(fig)
    return hist


'''
Helper function
'''   #返回一个列表，内容是文件夹下的所有文件夹名
def list_scenes(folder, extension):
    listFiles = []
    for f in glob.glob(os.path.join(folder, "*.mp4")):
        fname = os.path.splitext(f)[0]
        # print(fname)
        # Find the folders for each video
        if os.path.exists(fname):
            # Add the list of scenes
            listFiles += glob.glob(os.path.join(fname, "*."+extension))

    return listFiles


'''
Stores information for all scenes in a folder into json files
informations : scene length, color (feature)
'''
def store_color_features(folder):     
    listFiles = list_scenes(folder, "mp4")
    # print(listFiles)
    for f in listFiles:
        # extract_feature(video file, save thumbnail for not)
        # sceneLength 表示视频长度，hist 表示视频的颜色分布直方
        sceneLength, hist = extract_feature(f, False)     
        if hist.size > 0 :    # 如果子视频符合要求
            jsonpath = os.path.splitext(f)[0]+'.json'
            jsondata = {'length':sceneLength,'color':hist.tolist()}
            with open(jsonpath, 'w') as outfile:
                json.dump(jsondata, outfile)   # 保存颜色分布信息到.json文件

'''
Draws on plots each cluster to easily visualize the clustering results
'''   #可视化
def display_clusters(df):
    clusNum = -1
    columns = 10
    rows = 0
    totalImgCluster = 0
    indexImgCluster = 0

    for index, row in df.iterrows():
        # new cluster = new plot
        if row['cluster'] != clusNum:

            # Show previous plot
            if clusNum >= 0 :
                if totalImgCluster>10:
                    figManager = plt.get_current_fig_manager()
                    figManager.window.showMaximized()
                    plt.show()
                plt.close(fig)

            # Build new plot
            clusNum = row['cluster']
            fig = plt.figure()
            totalImgCluster = len(df[df['cluster'] == clusNum])
            print("-- Cluster n°%d containing %d videos --"%(clusNum,totalImgCluster))
            indexImgCluster = 0

            # Have 10 columns and adapted number lines
            rows = totalImgCluster//10 if totalImgCluster%10==0 else totalImgCluster//10 + 1

        # draw image
        indexImgCluster += 1
        plt.subplot(rows, columns, indexImgCluster)
        plt.axis("off")
        thumbPath = os.path.splitext(THUMBNAILS_PATH+os.path.basename(row['file']))[0]+".mp4.jpg"
        image = cv2.imread(thumbPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    plt.close(fig)


'''
Executes Kmeans on a given list of files
'''
def compute_kmeans(listFiles):
    # Read the features from json files
    arrHist = []
    arrLength = []
    # print(listFiles)
    for jsonFile in listFiles:
        with open(jsonFile, "r") as jFile:
            jdata = json.load(jFile)
            #print(arrHist)
            arrHist.append(jdata['color'])
            arrLength.append(jdata['length'])

    arrHist = np.array(arrHist)
    
    # Compute Kmeans on histograms to group videos
    kmeans = KMeans(n_clusters=CLUSTERS, init='random', random_state=2, max_iter=800, algorithm="full").fit(arrHist)
    
    # Display the clutering results
    df = pd.DataFrame.from_records(zip(listFiles,arrLength,kmeans.labels_), columns=['file','length','cluster'])
    df = df.sort_values(by=['cluster','file'])
    # df.to_csv("../statistics/clusters-%d.csv"%CLUSTERS)
    # display_clusters(df)
    return df   #返回一个dataframe，按照聚类标签排好序的
    

def main():             # TODO: 涉及体裁还没看
    # Extract the color features and store them
    store_color_features(sys.argv[1])

    # Kmeans for each style
    df = pd.read_csv(os.path.join(STATISTICS_PATH, "songs_on_server.csv"), sep=";")

    start = time.time()
    for resolution in ["16/9","40/17"]:
        for style in ["rock","pop","hiphop","electro"] :
            
            subDf = df.loc[(df["style"] == style) & (df["resolution"] == resolution)]
            if len(subDf) < MIN_VIDEOS_IN_STYLE:
                print('Skip KMeans for data with resolution: {}, style: {}'.format(resolution, style))
                continue
            
            print("Starting KMeans for style : %s"%style)
            listFiles = []

            # Get all scenes of MVs with same style
            for _, row in subDf.iterrows():
                listFiles += glob.glob(os.path.join(sys.argv[1], row["id"], "*.json"))

            kmeans = compute_kmeans(listFiles)
            kmeans.to_csv(os.path.join(STATISTICS_PATH, "kmeans_"+resolution[:2]+"_"+style+".csv"),index=False)

            print("Finished KMeans. Time elapsed : %f---------------\n"%(time.time()-start))
            start = time.time()

    for resolution in ["16/9","40/17"]:    
        print('Starting KMeans for resolution: {}'.format(resolution))
        subDf = df.loc[df["resolution"] == resolution]
        
        if resolution == '16/9' and len(subDf) == 0:
            # we just don't care about the 40/17 old-fashion videos whether it's given or not  
            raise Error('No videos with resolution 16/9 given')     
        listFiles = []

        # Get all scenes of MVs with same style
        for _, row in subDf.iterrows():
            listFiles += glob.glob(os.path.join(sys.argv[1], row["id"], "*.json"))

        kmeans = compute_kmeans(listFiles)
        kmeans.to_csv(os.path.join(STATISTICS_PATH, "kmeans_"+resolution[:2]+".csv"),index=False)
        print("Finished KMeans. Time elapsed : %f---------------\n"%(time.time()-start))
        start = time.time()

if __name__ == "__main__":
    main()
