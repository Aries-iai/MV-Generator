import random, os, json, time
import subprocess
import msaf
import types
from warnings import filterwarnings
import pandas as pd
import tempfile, shutil

from .music_recognition import get_music_infos, convert_genre_to_style
from .config import (
    BOUNDARY_OFFSET, END_OFFSET, CLUSTERS,
    STATISTICS_PATH, GENERATED_MV_PATH,
    NUM_CLUS_IN_VIDEO, AUTHORIZED_GENRES,
)


'''
Changes the order in which appear group of videos from one MV in cluster
'''
def shuffle_mvs(df):
    dic = {}
    
    # count the video clips number in each directory(source video which the clips sampled from)
    for index, row in df.iterrows():
        if os.path.dirname(row['file']) not in dic.keys():
            dic[os.path.dirname(row['file'])] = 1
        else:
            dic[os.path.dirname(row['file'])] += 1

    mvList = list(dic.keys())   # get the list of source videos used 
    random.shuffle(mvList)
    res = []
    for mv in mvList:
        res.append(df[df['file'].str.startswith(mv)])
    return pd.concat(res).reset_index(drop=True)


def assemble_videos(df, boundaries, tempDir):
    """
    assemble some videos using the result of df around the boundaries

    Args:
        df (pandas.Dataframe): result of videos clustering
            (e.g.   file,         length,     cluster id 
                    XXX.json      3.303300    0           )
        boundaries (numpy.array): list of timestamps in seconds
        tempDir (str): ...

    Returns:
        int: None or -1
    """
    # Very often msaf detects 2 times the last boundary. Remove it if necessary
    if boundaries[-1]-boundaries[-2]<20:
        boundaries = boundaries[1:-1]
    else:
        boundaries = boundaries[1:]

    # Get NUM_CLUS_IN_VIDEO random clusters such as sum lengths more than whole music
    # with approximately same proportion of each

    # List of number of scenes and total length for each cluster
    # print(CLUSTERS)

    # clusterLengths(list[tuple]): list of clusterLength
    # clusterLength (tuple): 
    #   (cluster id i, number of video clips in cluster i, total length of video clips in cluster i)
    clusLengths = [(i,len(df[df['cluster'] == i]), df[df['cluster']==i]['length'].sum()) for i in range(CLUSTERS)]
    # print(clusLengths)
    
    limit = boundaries[-1]/NUM_CLUS_IN_VIDEO - 10
    # print(limit)
    clusLengths = [x for x in clusLengths if x[2]>limit]
    #print(clusLengths,NUM_CLUS_IN_VIDEO)
    selectedClus = random.sample(clusLengths,NUM_CLUS_IN_VIDEO)
    limit = 0
    while sum([x[2] for x in selectedClus])<boundaries[-1] and limit<1000:
        selectedClus = random.sample(clusLengths,NUM_CLUS_IN_VIDEO)
        limit += 1
    if limit == 1000:
        print("Could not find clusters covering the whole video.")
        return -1

    # Append videos of selected 5 clusters
    videoLength = 0
    numBound = 1
    numClus = 0

    clus = [shuffle_mvs(df[df['cluster'] == selectedClus[i][0]]) for i in range(NUM_CLUS_IN_VIDEO)] # files in each cluster
    indexes = [0]*NUM_CLUS_IN_VIDEO # keeps track if last used file index for each cluster
    print(len(boundaries)-1)
    with open('video_structure.txt','w') as vidList:
        while videoLength < boundaries[-1]-END_OFFSET:
            print(numBound)
            # When reaching next boundary
            while numBound < len(boundaries) and videoLength < boundaries[numBound]-BOUNDARY_OFFSET:

                if indexes[numClus] < len(clus[numClus]) :
                    filename = os.path.splitext(clus[numClus].iloc[indexes[numClus]]['file'])[0]
                    fileLength = clus[numClus].iloc[indexes[numClus]]['length']

                    if videoLength+fileLength>boundaries[-1]-END_OFFSET: # stop this algorithm for the last END_OFFSET seconds
                        break
                    
                    if os.path.exists(tempDir+os.path.basename(filename)+'.mp4'):
                        indexes[numClus]+=1
                        continue
                        
                    # If going over boundary, cut scene
                    if videoLength + fileLength > boundaries[numBound]-BOUNDARY_OFFSET :
                        fileLength = boundaries[numBound]-BOUNDARY_OFFSET-videoLength
                        subprocess.call(['ffmpeg', '-loglevel', 'error', '-t', str(fileLength), '-i', filename+'.mp4', '-q', '0', tempDir+os.path.basename(filename)+'.mp4'])
                    else :
                        subprocess.call(['ffmpeg', '-loglevel', 'error', '-i', filename+'.mp4', '-q', '0', tempDir+os.path.basename(filename)+'.mp4'])
                
                    vidList.write("file '"+tempDir+os.path.basename(filename)+".mp4'\n")
                    videoLength += fileLength
                    indexes[numClus]+=1
                    
                else:
                    numClus = (numClus+1)%NUM_CLUS_IN_VIDEO

            # next boundary
            if numBound<len(boundaries)-1:
                numClus = (numClus+1)%NUM_CLUS_IN_VIDEO
                numBound += 1

            else:
                break

        # find only one video to end for the last seconds
        indexes[numClus] %= len(clus[numClus])
        while fileLength < boundaries[-1]-videoLength and indexes != [0]*NUM_CLUS_IN_VIDEO:
            if indexes[numClus] == 0: # if reached end of cluster, go to next cluster
                numClus = (numClus+1)%NUM_CLUS_IN_VIDEO
                indexes[numClus] %= len(clus[numClus])
            else:
                fileLength = clus[numClus].iloc[indexes[numClus]]['length']
                filename = os.path.splitext(clus[numClus].iloc[indexes[numClus]]['file'])[0]
                indexes[numClus] = (indexes[numClus]+1)%len(clus[numClus])
        
        if indexes == [0]*NUM_CLUS_IN_VIDEO: # not found a video long enough -> no fade
            subprocess.call(['ffmpeg', '-y', '-loglevel', 'error', '-i', filename+'.mp4', '-q', '0', tempDir+os.path.basename(filename)+'.mp4'])
        else: # fade out last 2 sec
            print("Add fading effect.")
            subprocess.call(['ffmpeg', '-y', '-loglevel', 'error', '-i', filename+'.mp4', '-q', '0',
            '-vf', 'fade=t=out:st='+str(fileLength-2)+':d=2', tempDir+os.path.basename(filename)+'.mp4'])

        vidList.write("file '"+tempDir+os.path.basename(filename)+".mp4'\n")


'''
Generator function for printing out progress info
'''
def log_progress():
    while True:
        progress = yield
        print(progress)


'''
Main steps for building the mv
args : input, output, video genre (optionnal)
callback.send : function that gives feedback to user
'''
def main(args, callback=log_progress()):
    if isinstance(callback, types.GeneratorType):
        next(callback)

    start = time.time()
    print('Analyzing music %s...'%args.input)

    if not os.path.exists(args.input):
        raise FileNotFoundError
    elif not os.path.exists(os.path.join(STATISTICS_PATH, 'kmeans_16.csv')) and \
         not os.path.exists(os.path.join(STATISTICS_PATH, 'kmeans_40.csv')):
        raise FileNotFoundError

    # 1. Get major changes in music
    callback.send('(1/3) Identifying significant rythm changes in music...\n This will take about a minute.')

    filterwarnings('ignore')
    # print(args.input)
    boundaries, labels = msaf.process(args.input,boundaries_id='cnmf')#, #boundaries_id='olda')
    if boundaries[-1] < 60 or boundaries[-1]>400:
        callback.send('Error : Please chose a music lasting between 60 and 400 seconds for getting a quality MV.')
        return -1
        
    callback.send('Key changes found at \n(%s) seconds\n'%' , '.join(map('{:.2f}'.format, boundaries)))

    # if args.data[-4:] == '.csv':
        # 2. Find music genre and style (music video style = larger category of genre)
    musicGenre = args.genre
    musicStyle = ''
    if musicGenre == '': # No genre given, must find it

        title, artist, musicGenre, musicStyle = get_music_infos(args.input)

        if musicStyle == '':
            callback.send('GenreWarning : The algorithm did not manage to recognize the music genre.\n'
                          'All the video clips will be used, or manually add genre with the argument --genre <name of genre> \n'
                          'with genre in ('+','.join(AUTHORIZED_GENRES)+').')
        else:
            callback.send('Music genre identified : %s.'%musicGenre)

    else:
        musicStyle = convert_genre_to_style(musicGenre)
        if musicStyle == '':
            callback.send('GenreError : This genre is not authorized. Please input one of the following ('+\
            ','.join(AUTHORIZED_GENRES)+') or let the algorithm find the genre.')
            return -1


    # 3. With the music genre, find appropriate videos in database
    callback.send('(2/3) Fetching matching videos in database...\n')
    
    # use k-means clustering result on scenes extracted from Music Videos with same genre and chose one resolution
    target_csv_file = None
    target_csv_format = 'kmeans_{}{}.csv'
    if len(musicStyle) > 0:
        target_csv_file = target_csv_format.format(args.resolution[:2], '_'+musicStyle)
    if not target_csv_file or \
       not os.path.exists(os.path.join(STATISTICS_PATH, target_csv_file)):
        target_csv_file = target_csv_format.format(args.resolution[:2], '')  

    clusterResult = pd.read_csv(os.path.join(STATISTICS_PATH, target_csv_file))


    # else:
        # use k-means clustering result on scenes extracted from Music Videos with same genre
        # listFiles = list_scenes(args.data,'json')
        # callback.send('(2/3) Generating K-Means for the database...')
        # clusterResult = compute_kmeans(listFiles)

    # 4. Join music scenes while respecting the clustering and the input music rythm
    callback.send('(3/3) Building the music video around these boundaries...\n This won \'t take long.\n')

    # Select and order videos for music clip
    tempDir = tempfile.mkdtemp('_music_video_build')+'/'
    print("Building the video file in folder %s"%tempDir)
    assemble_videos(clusterResult, boundaries, tempDir)

    # Concatenate videos
    subprocess.call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'video_structure.txt',
    '-c', 'copy', '-an', tempDir+'temp_video.mp4'])

    # Put input music on top of resulting video
    extension = args.extension
    args.output = os.path.join(
        GENERATED_MV_PATH, 
        os.path.splitext(os.path.basename(args.input))[0]+extension
    )
    # if extension != '.avi' and extension != '.mkv':
    #     args.output = os.path.splitext(args.output)[0]+'.mp4'
    #     if extension != '.mp4' :
    #         print('No format within (avi,mkv,mp4) given. Using default mp4 ...')

    # copies video stream and replace audio of arg 0 by arg 1
    subprocess.call(['ffmpeg', '-y', '-loglevel', 'error', '-i', tempDir+'temp_video.mp4', '-i', args.input,
    '-c:v' ,'copy', '-map', '0:v:0', '-map', '1:a:0', args.output])

    print('Video file %s written.\n'%args.output)
    callback.send('--- Finished building the music video in %f seconds. ---'%(time.time()-start))

    # Delete temp files
    shutil.rmtree(tempDir)

    # move video sturcture text to output directory
    shutil.copyfile('video_structure.txt', os.path.splitext(args.output)[0]+'_structure.txt')
    os.remove('video_structure.txt')
    # file = r'c:\Users\86157\Desktop\music2mv\static\videos'
    # mp4 = args.output[0]
    # final_file = os.path.join(file, mp4)
    # cmd = "ffmpeg.exe"+ " -i " + args.output[0] + ".mp4" + " -vcodec h264 -threads 5 -preset ultrafast " + final_file + ".mp4"
    # os.system(cmd)

    # Copy video to folder generated
    # if os.path.exists(GENERATED_MV_PATH):
    #     shutil.copyfile(args.output, 'generatedmvs/'+time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())+'.mp4')

    if callback is not None and isinstance(callback, types.GeneratorType):  # Close the generator
        callback.close()


