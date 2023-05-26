from __future__ import annotations
import csv
import heapq
import numpy as np

learning_order = ["Chair",
                  "Mailbox",
                  "Plant",
                  "Telescope",
                  "Table",
                  "Stove",
                  "Piano",
                  "Trashcan",
                  "Bookshelf",
                  "Wheelbarrow",
                  "Harp",
                  "Well"]


def load_objects(file_name):
    objects = {}
    with open(file_name, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            obj = (row[0], int(row[1]), int(row[2]))
            objects[obj[0]] = (obj[1], obj[2])

    print("ShortcutMap: Loaded {} objects".format(len(objects)))
    return objects


def transform_walls(input_file, output_file):
    with open(input_file, "r") as input_csv:
        reader = csv.reader(input_csv)
        next(reader)  # Skip the header row

        with open(output_file, "w", newline="") as output_csv:
            writer = csv.writer(output_csv)
            writer.writerow(["X", "Y"])  # Write header row to output file

            for row in reader:
                x, y = int(row[0]), int(row[1])
                new_x, new_y = x - 1, y - 1
                writer.writerow([new_x, new_y])


def get_tile_map(x_points, y_points):
    tile_map = [[np.zeros(2) for _ in range(len(x_points) - 1)] for _ in range(len(y_points) - 1)]
    for col in range(len(x_points) - 1):
        for row in range(len(y_points) - 1):
            # get the center point of the tile
            x = (x_points[col] + x_points[col + 1]) / 2
            y = (y_points[row] + y_points[row + 1]) / 2
            tile_map[row][col] = np.array([x, y])
    return tile_map


def generate_connectivity_matrix(grid_x, grid_y, tile_map):
    width = len(grid_x) - 1
    height = len(grid_y) - 1
    matrix = np.zeros((width * height, width * height))
    for row in range(len(grid_y) - 1):
        for col in range(len(grid_x) - 1):
            current_index = row * width + col
            current_center = tile_map[row][col]
            if row > 0:  # connect to block above
                above_index = (row - 1) * width + col
                above_center = tile_map[row - 1][col]
                matrix[current_index][above_index] = np.linalg.norm(current_center - above_center)
            if row < width - 1:  # connect to block below
                below_index = (row + 1) * width + col
                below_center = tile_map[row + 1][col]
                matrix[current_index][below_index] = np.linalg.norm(current_center - below_center)
            if col > 0:  # connect to block to the left
                left_index = row * width + (col - 1)
                left_center = tile_map[row][col - 1]
                matrix[current_index][left_index] = np.linalg.norm(current_center - left_center)
            if col < width - 1:  # connect to block to the right
                right_index = row * width + (col + 1)
                right_center = tile_map[row][col + 1]
                matrix[current_index][right_index] = np.linalg.norm(current_center - right_center)
            # connect to current block
            matrix[current_index][current_index] = 999
    return matrix


class ShortcutMap:
    def __init__(self, wall_file, object_file,
                 shortcut_output_file="shortcut_paths.csv",
                 learning_output_file="learning_paths.csv",
                 reverse_learning_output_file="reverse_learning_paths.csv",
                 learning=False
                 ):
        self.map_width = 7
        self.map_height = 7
        self.grid_x = [-3.7, -2.35, -1.25, -0.49, 0.4, 1.2, 2.0, 3.05]
        self.grid_y = [-3.7, -2.49, -1.49, -0.49, 0.49, 1.35, 2.10, 3.7]
        self.tile_map = get_tile_map(self.grid_x, self.grid_y)
        self.connectivity_matrix = generate_connectivity_matrix(self.grid_x, self.grid_y, self.tile_map)
        self.load_walls(wall_file)
        self.objects = load_objects(object_file)
        self.shortest_paths = self.calculate_shortest_paths()
        self.learning_paths = None
        self.reverse_learning_paths = None
        if learning:
            self.learning_paths = self.calculate_learning_path(learning_order)
            self.reverse_learning_paths = self.calculate_learning_path(learning_order[::-1])
            self.save_shortest_distances(learning_output_file, self.learning_paths)
            self.save_shortest_distances(reverse_learning_output_file, self.reverse_learning_paths)
        else:
            self.save_shortest_distances(shortcut_output_file, self.shortest_paths)

    def calculate_learning_path(self, order):
        # get path for consecutive objects and handle the last object to first object case
        path = dict()
        for i in range(len(order)):
            path[order[i]] = dict()

        for i in range(len(order)):
            for j in range(len(order)):
                if i != j:
                    if i < j:
                        seq = order[i:j + 1]
                    else:
                        seq = order[i:] + order[:j + 1]

                    new_path = []
                    for k in range(len(seq) - 1):
                        # concatenate the paths
                        new_path += self.get_shortest_path(seq[k], seq[k + 1])[1]
                    # remove duplicates in new_path
                    new_path = list(dict.fromkeys(new_path))

                    path[order[i]][order[j]] = (len(new_path), new_path)
        return path

    def get_learning_path(self, start_object, end_object):
        return self.learning_paths[start_object][end_object]

    def get_reverse_learning_path(self, start_object, end_object):
        return self.reverse_learning_paths[start_object][end_object]

    def get_shortest_path(self, start_object, end_object):
        try:
            return self.shortest_paths[start_object][end_object]
        except KeyError:
            pass
        try:
            distance, path = self.shortest_paths[end_object][start_object]
            return distance, path[::-1]
        except KeyError:
            return None

    def get_shortest_path_in_tiles(self, start_object, end_object):
        distance, path = self.get_shortest_path(start_object, end_object)
        if distance is None:
            return None
        return distance, [self.get_tile_center(pos) for pos in path]

    def convert_to_tile_path(self, path):
        return [self.get_tile_center(pos) for pos in path]

    def get_shortest_distance_in_tiles(self, start_object, end_object):
        path = self.get_shortest_path_in_tiles(start_object, end_object)
        # numpy norm
        distance = 0
        for i in range(len(path) - 1):
            distance += np.linalg.norm(path[i] - path[i + 1])
        return distance

    def get_shortest_distance(self, start_object, end_object):
        return self.get_shortest_path(start_object, end_object)[0]

    def get_tile_center(self, pos):
        return self.tile_map[pos[1]][pos[0]]

    def load_walls(self, file_name):
        with open(file_name, "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                x, y = int(row[0]), int(row[1])
                current_index = self.coord_to_index((x, y))
                t = self.index_to_coord(current_index)
                # print("X: {}, Y: {}, Index: {}, t: {}".format(x, y, current_index, t))
                # set current block and neighboring blocks to 0
                self.connectivity_matrix[current_index][current_index] = 0
                if x > 0:  # set block left to 0
                    above_index = self.coord_to_index((x - 1, y))
                    self.connectivity_matrix[current_index][above_index] = 0
                    self.connectivity_matrix[above_index][current_index] = 0
                if x < self.map_width - 1:  # set block right to 0
                    below_index = self.coord_to_index((x + 1, y))
                    self.connectivity_matrix[current_index][below_index] = 0
                    self.connectivity_matrix[below_index][current_index] = 0
                if y > 0:  # set block to the left to 0
                    left_index = self.coord_to_index((x, y - 1))
                    self.connectivity_matrix[current_index][left_index] = 0
                    self.connectivity_matrix[left_index][current_index] = 0
                if y < self.map_height - 1:  # set block to the right to 0
                    right_index = self.coord_to_index((x, y + 1))
                    self.connectivity_matrix[current_index][right_index] = 0
                    self.connectivity_matrix[right_index][current_index] = 0

    def check_coord_is_wall(self, coord):
        index = self.coord_to_index(coord)
        return self.connectivity_matrix[index][index] == 0

    def coord_to_index(self, coord):
        x, y = coord
        if x >= self.map_width or x < 0 or y >= self.map_height or y < 0:
            raise ValueError("Coord out of bounds")
        return int(x + y * self.map_width)

    def index_to_coord(self, index):
        x = index % self.map_width
        y = (index - x) // self.map_width
        return x, y

    def dijkstra(self, start):
        n = len(self.connectivity_matrix)
        distances = [float('inf')] * n
        distances[start] = 0
        heap = [(0, start)]
        prev = [None] * n

        while heap:
            current_distance, current_node = heapq.heappop(heap)
            if current_distance > distances[current_node]:
                continue

            for neighbor in range(n):
                weight = self.connectivity_matrix[current_node][neighbor]
                if weight > 0:
                    distance = current_distance + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        prev[neighbor] = current_node
                        heapq.heappush(heap, (distance, neighbor))

        return distances, prev

    def reconstruct_path(self, prev, end, coord=False):
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        if coord:
            return [self.index_to_coord(index) for index in path]
        return path

    def calculate_shortest_paths(self):
        shortest_paths = {}
        names = list(self.objects.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                start_name = names[i]
                end_name = names[j]
                start_obj = self.objects[start_name]
                end_obj = self.objects[end_name]
                start = (start_obj[0], start_obj[1])
                end = (end_obj[0], end_obj[1])
                # print("Calculating path from {} to {}".format(start, end))

                start_index = self.coord_to_index(start)
                end_index = self.coord_to_index(end)

                distances, prev = self.dijkstra(start_index)
                # add 1 to all distances to account for the fact that the distance is the number of blocks

                # distances = [distance + 1 for distance in distances]

                distance = distances[end_index]
                path = self.reconstruct_path(prev, end_index, coord=True)

                nested_dict = {end_name: (distance, path)}
                if start_name in shortest_paths:
                    shortest_paths[start_name].update(nested_dict)
                else:
                    shortest_paths[start_name] = nested_dict

        return shortest_paths

    def get_object_pos(self, name):
        return self.objects[name]

    def get_all_objects_position(self):
        pos = []
        for obj in self.objects:
            pos.append(self.objects[obj])
        return pos

    def get_traversability_matrix(self):
        # create a grid map where all the walls are 0 and all the free space is 1
        traversability_matrix = np.ones((self.map_width, self.map_height))
        for x in range(self.map_width):
            for y in range(self.map_height):
                if self.check_coord_is_wall((x, y)):
                    traversability_matrix[x, y] = 0

        # add walls around the map
        # traversability_matrix = np.pad(traversability_matrix, pad_width=1, mode='constant', constant_values=0)

        return traversability_matrix

    def get_shortest_path_from_two_coords(self, coord1, coord2):
        index1 = self.coord_to_index(coord1)
        index2 = self.coord_to_index(coord2)
        distance, prev = self.dijkstra(index1)
        path = self.reconstruct_path(prev, index2, coord=True)
        return path

    def save_shortest_distances(self, file_name, paths):
        with open(file_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["Object1", "Object1_X", "Object1_Y", "Object2", "Object2_X", "Object2_Y", "Shortest_Distance", "Path"])

            for start_name in paths.keys():
                for end_name in paths[start_name].keys():
                    start_obj = self.objects[start_name]
                    end_obj = self.objects[end_name]
                    shortest_distance, path = paths[start_name][end_name]
                    path = "->".join([str(coord) for coord in path])
                    writer.writerow(
                        [start_name, start_obj[0], start_obj[1], end_name, end_obj[0], end_obj[1],
                         shortest_distance, path])

    def save_survey_map(self):
        x_points = self.grid_x
        y_points = self.grid_y
        tile_map = [[100 for _ in range(len(x_points) - 1)] for _ in range(len(y_points) - 1)]
        for col in range(len(x_points) - 1):
            for row in range(len(y_points) - 1):
                if not self.check_coord_is_wall((col, row)):
                    # get the center point of the tile
                    w = abs(x_points[col] - x_points[col + 1])
                    h = abs(y_points[row] - y_points[row + 1])
                    area = w * h
                    tile_map[row][col] = 1

        resized_map = [[120] * (len(x_points) + 1) for _ in range(len(y_points) + 1)]
        for i in range(len(x_points) - 1):
            for j in range(len(y_points) - 1):
                resized_map[i + 1][j + 1] = tile_map[i][j]

        # convert resize_map to numpy array
        resized_map = np.array(resized_map)

        with open("survey_map.txt", "w", newline="") as csvfile:
            for i in range(len(x_points) + 1):
                for j in range(len(y_points) + 1):
                    csvfile.write(f"{i + 1}\t{j + 1}\t{resized_map[i][j]}\n")

        with open("landmarks_on_survey_map.txt", "w", newline="") as csvfile:
            for obj in self.objects:
                pos = self.objects[obj]
                csvfile.write(f"{obj},{pos[1] + 2},{pos[0] + 2}\n")
            pass
        # show the map
        # import matplotlib.pyplot as plt
        # plt.matshow(resized_map, cmap='gray')
        # plt.show()


def print_matrix(matrix):
    for row in matrix:
        print(row)


if __name__ == "__main__":
    xgrid = [-3.7, -2.35, -1.25, -0.49, 0.4, 1.2, 2.0, 3.05]
    ygrid = [-3.7, -2.49, -1.49, -0.49, 0.49, 1.35, 2.10, 3.7]
