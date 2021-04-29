import cv2
import shutil
import os

VIDEO_PATH = "./movie/IMG_3628.MOV"
EXTRACT_FOLDER = "./img/"
EXTRACT_FREQUENCY = 1


def extract_frames(video_path, dst_folder, index):
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        exit(1)
    count = 1
    while True:
        _, frame = video.read()
        if frame is None:
            break
        if count % EXTRACT_FREQUENCY == 0:
            save_path = "{}/{}.jpg".format(dst_folder, index)
            cv2.imwrite(save_path, frame)
            index += 1
        count += 1
    video.release()

    print("Totally save {:d} pics".format(index - 1))


def main():
    try:
        shutil.rmtree(EXTRACT_FOLDER)
    except OSError:
        pass

    os.mkdir(EXTRACT_FOLDER)
    # 抽取帧图片，并保存到指定路径
    extract_frames(VIDEO_PATH, EXTRACT_FOLDER, 1)


if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    print("fps = {}".format(fps))
    main()
