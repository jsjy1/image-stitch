import numpy as np
import cv2

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


class App:
    def __init__(self, video, track_len=10, detect_interval=1, max_points=5000, img_size=(1200, 800)):  # 构造方法，初始化一些参数和视频路径
        self.video = cv2.VideoCapture(video)
        self.isImgList = type(video) == list
        self.track_len = track_len
        self.detect_interval = detect_interval
        self.max_points = max_points
        self.img_size = img_size
        self.frame_idx = 0
        self.tracks = []

    def run(self):  # 光流运行方法
        # while True:
        while cv2.waitKey(1)!=ord('q'):
            if self.isImgList:
                ret, frame = self.frame_idx < len(self.video), self.video[self.frame_idx]
            else:
                ret, frame = self.video.read()  # 读取视频帧
            if ret:
                frame = cv2.resize(frame, self.img_size)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
                vis = frame.copy()
                if len(self.tracks) > 0:  # 检测到角点后进行光流跟踪
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                           **lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None,
                                                            **lk_params)  # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # 得到角点回溯与前一帧实际角点的位置变化关系
                    good = d < 1  # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                    self.tracks, p0, p1 = [self.tracks[i] for i in range(len(good)) if good[i]], p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)
                    for tr, (x, y) in zip(self.tracks, p1):  # 将跟踪正确的点列入成功跟踪点
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False,
                                  (0, 255, 0))  # 以上一振角点为初始点，当前帧跟踪到的点为终点划线


                if self.frame_idx % self.detect_interval == 0:  # 每隔detect_interval帧检测一次特征点
                    mask = np.zeros_like(frame_gray)  # 初始化和视频大小相同的图像
                    mask[:] = 255  # 将mask赋值255也就是算全部图像的角点
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:  # 跟踪的角点画圆
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # 像素级别角点检测
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            if len(self.tracks) > self.max_points:
                                del self.tracks[0]
                            self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

                self.frame_idx += 1
                self.prev_gray = frame_gray
                # vis = cv2.resize(vis, (1200,800))
                cv2.putText(vis, str(self.frame_idx), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.imshow('lk_track', vis)
                print(self.frame_idx)
                cv2.waitKey(1)
            else:
                break
        self.video.release()

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0#"D:/Desktop/newdat/test.mp4"

    App(video_src).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
