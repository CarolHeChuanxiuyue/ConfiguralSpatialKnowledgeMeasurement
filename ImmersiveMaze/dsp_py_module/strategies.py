import numpy as np
import matplotlib.pyplot as plt


def survey_strategy(map, startX, startY, endX, endY, dispWave):
    GOAL = np.array([endX, endY])
    START = np.array([startX, startY])

    Ne1, Ne2 = map.shape
    SPIKE = 1
    REFRACTORY = -5
    W_INIT = 5
    LEARNING_RATE = 1.0

    wgt = np.zeros((Ne1, Ne2, Ne1, Ne2))  # weight matrix, D
    for i in range(Ne1):
        for j in range(Ne2):
            for m in range(-1, 2):
                for n in range(-1, 2):
                    if m * n == 0 and m != n and 0 <= i + m < Ne1 and 0 <= j + n < Ne2:
                        wgt[i, j, i + m, j + n] = W_INIT  # set weight to 5 for adjacent cells

    delayBuffer = np.zeros((Ne1, Ne2, Ne1, Ne2))  # delayBuffer for each neuron
    v = np.zeros((Ne1, Ne2))
    u = np.zeros((Ne1, Ne2))

    v[startX, startY] = SPIKE  # set start point to spike

    foundGoal = False
    timeSteps = 0
    aer = []  # address event representation

    while not foundGoal:
        timeSteps += 1
        fExcX, fExcY = np.where(v >= SPIKE)  # find all cells that spike
        aer.append(np.column_stack((timeSteps * np.ones(len(fExcX)), fExcX, fExcY)))  # add spike cells to aer

        if fExcX.size > 0:  # if there are spike cells
            for fExc_i, fExc_j in zip(fExcX, fExcY):
                u[fExc_i, fExc_j] = REFRACTORY
                wgt[fExc_i, fExc_j] = delay_rule(wgt[fExc_i, fExc_j], map[fExc_i, fExc_j], LEARNING_RATE)
                delayBuffer[fExc_i, fExc_j] = np.round(wgt[fExc_i, fExc_j])
                if fExc_i == endX and fExc_j == endY:  # if goal cell is activated, then it is found
                    foundGoal = True

        if not foundGoal:
            Iexc = u.copy()  # Input current
            for i in range(Ne1):
                for j in range(Ne2):
                    fExcX, fExcY = np.where(delayBuffer[:, :, i, j] == 1)  # find all cells that spike
                    if fExcX.size > 0:
                        for fExc_i, fExc_j in zip(fExcX, fExcY):
                            Iexc[i, j] += wgt[
                                              fExc_i, fExc_j, i, j] > 0  # if there is a spike, then add weight to input current
            v += Iexc
            u = np.minimum(u + 1, 0)

        delayBuffer = np.maximum(0, delayBuffer - 1)

    # print(START)
    # print(GOAL)
    path = get_path(np.vstack(aer), map, START, GOAL)

    if dispWave:
        path_len = len(path)
        map[path[0]['x'], path[0]['y']] = 75
        for i in range(1, path_len):
            map[path[i]['x'], path[i]['y']] = 20
        map[path[-1]['x'], path[-1]['y']] = 50
        ax, fig = plt.subplots()
        ax.imshow(map)
        ax.set_title(f"Survey:({startX},{startY}) :({endX},{endY})")
        fig.show()
        ax.clear()
        plt.close(fig)

    return path


def delay_rule(wBefore, value, learnRate):
    valMat = learnRate * (value - wBefore)  # update weight
    wAfter = wBefore + (wBefore > 0) * valMat  # update weight
    return wAfter


def get_path(spks, map, s, e):
    path = [{'x': e[0], 'y': e[1]}]

    while np.linalg.norm(np.array([path[0]['x'], path[0]['y']]) - s) > 1.0:  # while start point is not reached
        path = [{'x': e[0], 'y': e[1]}]

        for i in range(int(spks[-1, 0] - 1), 0, -1):
            inx = np.where(spks[:, 0] == i)[0]
            found = False
            k = 0
            lst = []

            for j in inx:
                if np.linalg.norm(np.array([path[0]['x'], path[0]['y']]) - spks[j, 1:]) < 1.5:
                    k += 1
                    lst.append({'x': int(spks[j, 1]), 'y': int(spks[j, 2])})
                    found = True

            if found:
                cost = np.finfo(np.float64).max
                # convert lst to integer array

                for m in range(k):
                    if map[int(lst[m]['x']), int(lst[m]['y'])] < cost:
                        cost = map[int(lst[m]['x']), int(lst[m]['y'])]
                        minx = m
                        dist = np.linalg.norm(s - np.array([lst[m]['x'], lst[m]['y']]))
                    elif map[lst[m]['x'], lst[m]['y']] == cost and np.linalg.norm(
                            s - np.array([lst[m]['x'], lst[m]['y']])) < dist:
                        minx = m
                        dist = np.linalg.norm(s - np.array([lst[m]['x'], lst[m]['y']]))

                path.insert(0, lst[minx])

    return path


import numpy as np


def topological_strategy(map, lm, start, sX, sY, end, eX, eY, shortcut=None):
    LARGE_NUM = 999999

    last = False
    path = []
    aX = sX
    aY = sY

    while not last:
        eu_dist_agent_to_goal = np.linalg.norm(np.array([aX, aY]) - np.array([eX, eY]))
        di_dist_agent_to_goal = shortcut.get_shortest_distance_in_tiles(start, end)
        min_dist_to_lm = LARGE_NUM

        for lm_id, lm_coords in lm.items():
            if lm_id == start or lm_id == end:
                continue
            eu_dist_to_lm = np.linalg.norm(lm_coords - np.array([aX, aY]))
            eu_dist_lm_to_goal = np.linalg.norm(lm_coords - np.array([eX, eY]))
            di_dist_to_lm = shortcut.get_shortest_distance_in_tiles(start, lm_id)
            di_dist_lm_to_goal = shortcut.get_shortest_distance_in_tiles(lm_id, end)

            if (eu_dist_to_lm < eu_dist_agent_to_goal and
                    eu_dist_lm_to_goal < eu_dist_agent_to_goal and
                    eu_dist_to_lm < min_dist_to_lm and
                    eu_dist_to_lm > 0 and
                    di_dist_to_lm < di_dist_agent_to_goal and
                    di_dist_to_lm < di_dist_lm_to_goal
            ):
                min_dist_to_lm = eu_dist_to_lm
                min_lm_idx = lm_id

        if min_dist_to_lm == LARGE_NUM:
            topoX = eX
            topoY = eY
        else:
            topoX = lm[min_lm_idx][0]
            topoY = lm[min_lm_idx][1]

        # print(f"Start{aX},{aY} End{eX},{eY} Topo{topoX},{topoY}")
        p_topo_seg = survey_strategy(map, aX, aY, topoX, topoY, 0)
        # print(f"SEG:{p_topo_seg}")
        if len(p_topo_seg) > 1:
            path = path + p_topo_seg[1:]
        else:
            path = path + p_topo_seg

        if topoX == eX and topoY == eY:
            last = True
        else:
            aX = topoX
            aY = topoY

    # path = [{'x': coord[0], 'y': coord[1]} for coord in path]
    path.insert(0, {'x': sX, 'y': sY})

    # print([(coord['x'], coord['y']) for coord in path])

    return path


def load_map(filename):
    # create a matrix of zeros with size 12x12
    matrix = np.zeros((9, 9), dtype=int)
    # read each line of the file
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            map_row = line.strip().split('\t')
            i = int(map_row[0]) - 1
            j = int(map_row[1]) - 1
            matrix[i][j] = int(float(map_row[2]))

    return matrix


def load_landmarks(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        landmarks = {}
        for line in lines:
            lm_row = line.strip().split(',')
            landmarks[lm_row[0]] = ([int(lm_row[1]) - 1, int(lm_row[2]) - 1])
        return landmarks
    pass


class Strategy:
    def __init__(self, map_path, landmark_path, shortcut_map=None):
        self.matrix = load_map(map_path)
        self.landmarks = load_landmarks(landmark_path)
        self.keys = list(self.landmarks.keys())
        self.shortcut_map = shortcut_map

        pass

    def get_path(self, start, end):
        sx, sy = self.landmarks[start]
        ex, ey = self.landmarks[end]
        path = topological_strategy(self.matrix.copy(), self.landmarks, start, sx, sy, end, ex, ey, self.shortcut_map)
        # return [[coord['x'] - 1, 7 - coord['y']] for coord in path]
        return [[coord['y'] - 1, coord['x'] - 1] for coord in path]

    def get_all_topological_plots(self, folder="topo_plot", save_only=False):
        keys = self.keys
        lm = self.landmarks
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                print(f"Plotting {keys[i]} to {keys[j]}")
                n1 = keys[i]
                n2 = keys[j]
                sx, sy = lm[keys[i]]
                ex, ey = lm[keys[j]]
                tm = self.matrix.copy()
                path = topological_strategy(tm, lm, n1, sx, sy, n2, ex, ey, self.shortcut_map)

                path_len = len(path)
                tm[path[0]['x']][path[0]['y']] = 75
                for x in range(1, path_len):
                    tm[path[x]['x']][path[x]['y']] = 20
                tm[path[-1]['x']][path[-1]['y']] = 50
                plt.close(plt.gcf())
                plt.matshow(tm)
                plt.title(f"Topological {n1}({sx},{sy}) {n2}({ex},{ey})")

                import os
                if not os.path.exists(folder):
                    os.makedirs(folder)

                plt.savefig(f"{folder}/{n1}_{n2}.png")
                if not save_only:
                    plt.show()

    def plot_path(self, start, end):
        sx, sy = self.landmarks[start]
        ex, ey = self.landmarks[end]
        tm = self.matrix.copy()
        path = topological_strategy(tm, self.landmarks, start, sx, sy, end, ex, ey, self.shortcut_map)
        path_len = len(path)
        tm[path[0]['x']][path[0]['y']] = 75
        for x in range(1, path_len):
            tm[path[x]['x']][path[x]['y']] = 20

        tm[path[-1]['x']][path[-1]['y']] = 50

        # for name in self.landmarks.keys():
        #     x, y = self.landmarks[name]
        #     tm[x][y] = 60
        plt.close(plt.gcf())
        plt.matshow(tm)
        plt.show()
