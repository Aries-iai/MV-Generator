import os

PWD = os.path.dirname(__file__) 

def get_src_path(*paths):
    return os.path.abspath(os.path.join(PWD, *paths))

# global params 

NUM_CLUS_IN_VIDEO = 30  # number of clusters of video clips to assemble MV 
MIN_VIDEOS_IN_STYLE = 10 # min number of videos in each style
CLUSTERS = 90   # number of clusters 

BOUNDARY_OFFSET = 0.50 # Delay in boundary delection
END_OFFSET = 3
RESOLUTION_PROBABILITY = 0.3333 # deprecated: Probability for resolution format (640x360 or 640x272)

AUTHORIZED_GENRES = ['alternative','metal','rock','pop','hip-hop','R&B','dance','techno','house','indie','electro']

ACRCLOUD_CONFIG_FILE = get_src_path('../.app_secret.json')
STATISTICS_PATH = get_src_path('../statistics/')
THUMBNAILS_PATH = get_src_path("../statistics/imgClusters/")
GENERATED_MV_PATH = get_src_path("../static/videos")

VIDEO_SPLIT_TEMPLATE = '$VIDEO_NAME-$SCENE_NUMBER'
FILE_SCENE_LENGH = get_src_path('../statistics/scenes_length.csv')
FILE_SCENE_NUMBER = get_src_path('../statistics/scenes_number.csv')