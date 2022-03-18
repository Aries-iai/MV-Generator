# mv_gen_web

## 1. 配置环境

### 环境
* *Python 3.7*
* ffmpeg

### Packages: 
* scipy 1.2
* librosa
* scenedetect[opencv-headless]
* msaf

可使用以下命令下载以上依赖：
```shell
sudo apt-get install ffmpeg
pip install -r requirements.txt
```

注意：若 `acrcloud_sdk_python` 下载失败，可使用以下命令：
```shell
git clone https://github.com/acrcloud/acrcloud_sdk_python
cd acrcloud_sdk_python/
python setup.py install
``` 

详见 `./requirements.txt`。

## 2. 下载视频素材
建议使用 `youtube-dl` 下载 https://www.youtube.com/watch?v=JGwWNGJdvx8&list=PLTC7VQ12-9raqhLCx1S1E_ic35t94dj28 中的 MV 合集（建议视频素材为音乐 MV）。

`youtube-dl` 安装方式详见 https://github.com/ytdl-org/youtube-dl。

建议将视频下载至 `./data` 路径下，并存储为 `.mp4` 格式。

## 3. 运行代码对视频素材进行预处理

- 计算并保存 `./data/` 目录下的视频素材信息
```shell
python mvgen/database_constitution.py ./data
``` 
- 视频分割
```shell
python mvgen/video_analysis.py ./data
```
- 提取视频特征
```shell
python mvgen/feature_color.py ./data
```

以上操作将提取到的视频信息存储到 `./statistics` 中，将临时文件存储到 `data` 中。


### 注意
已将 `acrcloud music` API 的 `key` 存储在 `.app_secret.json` 中。
若 API 失效，有可能 14 天试用期已到，请自行注册免费账号并重新填写 `.app_secret.json`。

## 4. 在网页端上传音乐使生成视频
```shell
python app.py
```
在http://127.0.0.1:5000/ 网页上点击选择文件，选择硬盘内的任意mp3，再点击上传，等待一分钟左右即会生成视频。
生成的视频在`static/videos`中，网页端的gallery也会显示生成的视频。



