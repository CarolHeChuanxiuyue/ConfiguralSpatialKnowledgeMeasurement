from __future__ import annotations
import csv
import heapq
import numpy as np


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
    def __init__(self, wall_file, object_file, shortest_distance_file):
        self.map_width = 7
        self.map_height = 7
        self.grid_x = [-3.7, -2.35, -1.25, -0.49, 0.4, 1.2, 2.0, 3.05]
        self.grid_y = [-3.7, -2.49, -1.49, -0.49, 0.49, 1.35, 2.10, 3.7]
        self.tile_map = get_tile_map(self.grid_x, self.grid_y)
        self.connectivity_matrix = generate_connectivity_matrix(self.grid_x, self.grid_y, self.tile_map)
        self.load_walls(wall_file)
        self.objects = load_objects(object_file)
        self.shortest_paths = self.calculate_shortest_paths()
        self.save_shortest_distances(shortest_distance_file)

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

    def save_shortest_distances(self, file_name):
        with open(file_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["Object1", "Object1_X", "Object1_Y", "Object2", "Object2_X", "Object2_Y", "Shortest_Distance", "Path"])

            for start_name in self.shortest_paths.keys():
                for end_name in self.shortest_paths[start_name].keys():
                    start_obj = self.objects[start_name]
                    end_obj = self.objects[end_name]
                    shortest_distance, path = self.shortest_paths[start_name][end_name]
                    path = "->".join([str(coord) for coord in path])
                    writer.writerow(
                        [start_name, start_obj[0], start_obj[1], end_name, end_obj[0], end_obj[1],
                         shortest_distance, path])


def print_matrix(matrix):
    for row in matrix:
        print(row)


if __name__ == "__main__":
    xgrid = [-3.7, -2.35, -1.25, -0.49, 0.4, 1.2, 2.0, 3.05]
    ygrid = [-3.7, -2.49, -1.49, -0.49, 0.49, 1.35, 2.10, 3.7]
    tilemap = get_tile_map(xgrid, ygrid)
    print_matrix(tilemap)
    mat = generate_connectivity_matrix(xgrid, ygrid)
    print_matrix(mat)
