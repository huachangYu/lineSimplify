"""
SimpiPoly算法的复现，没有参考任何开源代码，仅仅根据论文描述进行复现的.
完成人：余华昌
参考文献：
刘钊，高培超，施洪刚，等. 一种基于曲率的线状目标简化算法. 测绘科学. 2014,10. 39(10):106-109
Copyright 2019, Yu Huachang, CSU
This code may be freely used and distributed, so long as it maintains
this copyright line.
Version 1.0, Date 2019-5-19 15.03.
"""
import math
import numpy as np
import pandas as pd
import pylab as pl


def bezier(t, n=4, points=None):
    """
    这里仅实现了4阶的贝塞尔曲线,所以可以忽略n这个参数
    :param t: 自变量
    :param n: 只能为4，尚未实现其他次数的贝塞尔曲线
    :param points: 5个控制点,形式如[[x0,y0],[x1,y1],...[x4,y4]]
    :return: 
    """
    if points.shape[0] != n + 1:
        print("输入点号错误")
        return
    res = 0
    res = (points[0, :] * pow((1 - t), 4) + points[1, :] * 4 * t * pow(1 - t, 3)
           + points[2, :] * 6 * pow(t, 2) * pow(1 - t, 2) + points[3, :] * 4 * pow(t, 3) * (1 - t)
           + points[4, :] * pow(t, 4))
    return res


def bazier_diff_x(t, n=4, points=None):
    """
    对x的一阶导数
    :param t: 
    :param n: 
    :param points: 
    :return: 
    """
    p0 = points[0, 0]
    p1 = points[1, 0]
    p2 = points[2, 0]
    p3 = points[3, 0]
    p4 = points[4, 0]
    res = 4 * p0 * pow(t - 1, 3) - 4 * p1 * pow(t - 1, 3) - 4 * p3 * pow(t, 3) + 4 * p4 * pow(t, 3) - 12 * p1 * t * pow(
        t - 1, 2) + 12 * p2 * t * pow(t - 1, 2) - 12 * p3 * pow(t, 2) * (t - 1) + 6 * p2 * pow(t, 2) * (2 * t - 2)
    return res


def bazier_diff_y(t, n=4, points=None):
    """
    对y的一阶导数
    :param t: 
    :param n: 
    :param points: 
    :return: 
    """
    p0 = points[0, 1]
    p1 = points[1, 1]
    p2 = points[2, 1]
    p3 = points[3, 1]
    p4 = points[4, 1]
    res = 4 * p0 * pow(t - 1, 3) - 4 * p1 * pow(t - 1, 3) - 4 * p3 * pow(t, 3) + 4 * p4 * pow(t, 3) - 12 * p1 * t * pow(
        t - 1, 2) + 12 * p2 * t * pow(t - 1, 2) - 12 * p3 * pow(t, 2) * (t - 1) + 6 * p2 * pow(t, 2) * (2 * t - 2)
    return res


def bazier_diff2_x(t, n=4, points=None):
    """
    对x的二阶导数
    :param t: 
    :param n: 
    :param points: 
    :return: 
    """
    p0 = points[0, 0]
    p1 = points[1, 0]
    p2 = points[2, 0]
    p3 = points[3, 0]
    p4 = points[4, 0]
    res = (12 * p0 * pow(t - 1, 2) - 24 * p1 * pow(t - 1, 2) + 12 * p2 * pow(t - 1, 2) + 12 * p2 * pow(t, 2)
           - 24 * p3 * pow(t, 2) + 12 * p4 * pow(t, 2) - 24 * p3 * t * (t - 1) - 12 * p1 * t * (2 * t - 2)
           + 24 * p2 * t * (2 * t - 2))
    return res


def bazier_diff2_y(t, n=4, points=None):
    """
    对y的二阶导数
    :param t: 
    :param n: 
    :param points: 
    :return: 
    """
    p0 = points[0, 1]
    p1 = points[1, 1]
    p2 = points[2, 1]
    p3 = points[3, 1]
    p4 = points[4, 1]
    res = (12 * p0 * pow(t - 1, 2) - 24 * p1 * pow(t - 1, 2) + 12 * p2 * pow(t - 1, 2) + 12 * p2 * pow(t, 2)
           - 24 * p3 * pow(t, 2) + 12 * p4 * pow(t, 2) - 24 * p3 * t * (t - 1) - 12 * p1 * t * (2 * t - 2)
           + 24 * p2 * t * (2 * t - 2))
    return res


def curvature(t, n=4, points=None):
    """
    计算贝塞尔曲线的曲率
    :param t: 
    :param n: 
    :param points: 
    :return: 
    """
    x1 = bazier_diff_x(t, n, points)
    x2 = bazier_diff2_x(t, n, points)
    y1 = bazier_diff_y(t, n, points)
    y2 = bazier_diff2_y(t, n, points)
    m1 = abs(x1 * y2 - y1 * x2)
    m2 = pow(x1, 2) + pow(y1, 2)
    return m1 / pow(m2, 1.5)


def dlocal(pt0, pt2, pt1):
    """
    局部偏移量
    :param pt0: 
    :param pt2: 
    :param pt1: 
    :return: 
    """
    res1 = 1.0 * abs((pt0[0] - pt1[0]) * (pt1[1] - pt2[1]) - (pt0[1] - pt1[1]) * (pt1[0] - pt2[0]))
    res2 = math.sqrt(math.pow(pt1[0] - pt2[0], 2) + math.pow(pt1[1] - pt2[1], 2))
    return res1 / res2


def run():
    dis_thres = 0.3
    rd_thres = 0.3
    data_path = 'xy.csv'  # 点的数据
    data = pd.read_csv(data_path, index_col=0)
    line_points = data.values
    num = line_points.shape[0]
    rdsum = np.zeros(num)
    rdnum = np.zeros(num)
    rdave = np.zeros(num)
    dis = np.zeros(num)
    ipt = 0
    while ipt + 5 < num:
        points = line_points[ipt:ipt + 5, :]
        rdsum[ipt + 1] += curvature(0.25, points=points)
        rdnum[ipt + 1] += 1
        rdsum[ipt + 2] += curvature(0.5, points=points)
        rdnum[ipt + 2] += 1
        rdsum[ipt + 3] += curvature(0.75, points=points)
        rdnum[ipt + 3] += 1
        ipt = ipt + 1
    for ipt in range(1, num - 1):
        pt0 = line_points[ipt, :]
        pt1 = line_points[ipt - 1, :]
        pt2 = line_points[ipt + 1, :]
        dis[ipt] = dlocal(pt0, pt1, pt2)
        rdave[ipt] = rdsum[ipt] / rdnum[ipt]
    xs = []
    ys = []
    xs += [line_points[0, 0]]
    ys += [line_points[0, 1]]
    for ipt in range(1, num - 1):
        if not dis[ipt] < dis_thres or rdave[ipt] > rd_thres:
            xs += [line_points[ipt, 0]]
            ys += [line_points[ipt, 1]]
    xs += [line_points[num - 1, 0]]
    ys += [line_points[num - 1, 1]]
    print(rdave)
    print(dis)

    pl.rcParams['font.sans-serif'] = ['SimHei']
    pl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plot1, = pl.plot(line_points[:, 0], line_points[:, 1])
    # pl.plot(xs, ys)
    plot2,=pl.plot(line_points[:,0],rdave)
    pl.scatter(line_points[:,0],line_points[:,1],marker='*',c = 'r',s=4)
    pl.title('偏移阈值：' + str(dis_thres) + '  曲率阈值：' + str(rd_thres))
    pl.legend([plot1,plot2],("线目标","伪曲率"))
    print('原始点的总数量', num)
    print('剩余点的数量：', len(xs))

    pl.show()


if __name__ == '__main__':
    run()
