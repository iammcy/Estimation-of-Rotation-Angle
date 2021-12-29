import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt
import math
import os


def imread(img_path):
    img = cv2.imread(img_path)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img


def crop(img):
    h, w = img.shape[:2]
    x0 = w // 2 - 418 // 2 - 1
    y0 = h // 2 - 418 // 2 - 1
    img = img[y0 : y0+419, x0 : x0+419]

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img


def RGB2YCrCb(img):
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # cv2.imshow("img", img_YCrCb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img_YCrCb[:,:,0]


def Laplacian(img):
    # cv2.CV_64F 提高数据精度，避免负数梯度截断为0
    re = cv2.Laplacian(img, cv2.CV_16S)
    # 缩放，获取绝对值，转换为无符号的8位类型
    re = cv2.convertScaleAbs(re)

    # cv2.imshow("img", re)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return re


def FFT(img):
    ret = []
    for i in range(len(img)):
        fft = np.fft.fft(img[i])
        fftshift = np.fft.fftshift(fft)
        amp = np.abs(fftshift)
        ret.append(amp)
    return np.array(ret)


def Peak_Counting(dft, delta=5):
    n = len(dft[0])
    cnt = np.zeros(n)
    for i in range(len(dft)):
        for j in range(n):
            left = j-delta if j-delta > 0 else 0
            right = j+delta if j+delta < n else n-1
            if dft[i][j] == dft[i][left : right + 1].max():
                cnt[j] += 1
    freq = np.fft.fftshift(np.fft.fftfreq(n=n,d=1))
    return cnt, freq


def plot(cnt, freq):
    plt.figure()
    plt.plot(freq, cnt)
    plt.ylim(0, len(freq) * 2)
    plt.xlabel("Frequency")
    plt.ylabel("Counts")
    plt.show()


def Peak_Detection(cnt, freq,  W=5, T=2):
    # plot(cnt, freq)
    candidate = []
    for i in range(len(cnt)):
        left = i-(W-1)//2 if i-(W-1)//2 > 0 else 0
        right = i+(W-1)//2 if i+(W-1)//2 < len(cnt) else len(cnt)-1
        if cnt[i] - np.median(cnt[left : right+1]) >= T:
            candidate.append(i)
    if len(candidate) == 0:
        print("not rotated")
        return

    # to select the largest two peaks
    candidate = np.array(candidate)
    cnt_can = cnt[candidate]
    freq_can = freq[candidate]
    sorted_index = cnt_can.argsort()
    cnt_can = cnt_can[sorted_index]
    freq_can = freq_can[sorted_index]
    # print(cnt_can)
    # print(freq_can)

    ret = []
    ret.append((cnt_can[-1], freq_can[-1]))
    ret.append((cnt_can[-2], freq_can[-2]))

    return ret


def cal_angle(data):
    cnt_1, freq_1 = data[0]
    cnt_2, freq_2 = data[1]
    angle_1 = []
    angle_2 = []

    # for freq_1
    angle = math.degrees(math.acos(1-freq_1))
    angle_1.append(angle)
    angle = math.degrees(math.acos(freq_1))
    angle_1.append(angle)
    angle = math.degrees(math.asin(freq_1))
    angle_1.append(angle)
    angle = math.degrees(math.asin(1-freq_1))
    angle_1.append(angle)

    # for freq_2
    angle = math.degrees(math.acos(1-freq_2))
    angle_2.append(angle)
    angle = math.degrees(math.acos(freq_2))
    angle_2.append(angle)
    angle = math.degrees(math.asin(freq_2))
    angle_2.append(angle)
    angle = math.degrees(math.asin(1-freq_2))
    angle_2.append(angle)

    # 1. to select the overlap angle
    # ret = []
    # angle_1 = np.array(angle_1)
    # angle_2 = np.array(angle_2)
    # overlap = np.intersect1d(np.round(angle_1), np.round(angle_2))
    # for angle in overlap:
    #     ang = (angle_1[np.where(np.round(angle_1) == angle)] + angle_2[np.where(np.round(angle_2) == angle)]) / 2
    #     ret.append(ang)
    # ret = np.round(ret, 3).reshape(1, -1)[0]

    # 2. return all angle
    ret = angle_1 + angle_2

    return ret


def main(img_path):
    # 1. 读取旋转后的测试图片
    img = imread(img_path)
    # 2. crop步骤
    img = crop(img)
    # 3. 提取图像亮度分量
    img_Y = RGB2YCrCb(img)
    # 4. 拉普拉斯算子提取边缘
    edge_map = Laplacian(img_Y)
    # 5. 对每一行进行FFT
    dft = FFT(edge_map)
    # 6. 尖峰统计
    cnt, freq = Peak_Counting(dft)
    # 7. 尖峰检测
    n = len(cnt)
    largest = Peak_Detection(cnt[n//2+1 : -1], freq[n//2+1 : -1])
    # 8. 角度计算
    result = cal_angle(largest)
    return result


def evaluate(img_dir):
    xlist = np.array([1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    # acc list
    ylist = np.zeros(len(xlist))
    cntlist = np.zeros(len(xlist))
    # 210(张图片) * 11(个旋转角度) 的测试结果 
    # ylist = np.array([0.62857143, 0.77142857, 0.76666667, 0.81904762, 0.81904762, 0.9047619, 0.91428571, 0.80952381, 0.97142857, 0.98095238, 0.98095238]) * 100

    for idx, img_name in enumerate(os.listdir(img_dir)):
        angle = int(img_name.split('_')[-1].split('.')[0])
        result = main(os.path.join(img_dir, img_name))
        result = np.array(result)
        error = np.abs(result - angle)

        cntlist[int(np.where(xlist == angle)[0])] += 1
        if error.min() < 0.5:
            ylist[int(np.where(xlist == angle)[0])] += 1

        if (idx + 1) % 200 == 0:
            print("---------the {} img---------".format(idx+1))
            print("acc number: ", ylist)
            print("total number: ", cntlist)

    ylist = ylist / cntlist * 100

    print("--------------final-------------")
    print("acc: ", ylist)

    plt.figure()
    plt.plot(xlist, ylist)
    plt.scatter(xlist, ylist, alpha = 2/5)
    plt.ylim(0, 100)
    plt.xticks(xlist)
    plt.xlabel("Angle in degrees")
    plt.ylabel("C(%)")
    plt.show()


if __name__ == '__main__':
    # 配置程序参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path')
    parser.add_argument('--img_dir')
    args = parser.parse_args()
    print(args)

    # 主程序入口
    result = main(args.img_path)
    # 论文例图结果角度：14.9度
    print(result)

    # 批处理评估
    if args.img_dir is not None:
        evaluate(args.img_dir)
    