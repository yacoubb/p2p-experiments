import cv2
import numpy as np
from scipy.spatial.distance import cdist
from colorsys import hsv_to_rgb


class NetworkDrawer(object):
    def __init__(self):
        self.peers = []
        self.positions = np.zeros((0, 2))
        self.velocities = np.zeros((0, 2))

        self.camera_coords = [0, 0, 0, 0]
        self.count = 0

    def step(self, peers):
        for diff in set(self.peers).symmetric_difference(set(peers)):
            # print(diff)
            if diff not in self.peers:
                # addition
                # print(f"adding {diff.index}")

                self.peers.append(diff)
                new_positions = np.zeros((len(self.peers), 2))
                new_positions[:-1, :] = self.positions
                new_positions[-1, :] = np.random.normal(size=(2))
                self.positions = new_positions

                new_velocities = np.zeros((len(self.peers), 2))
                new_velocities[:-1, :] = self.velocities
                new_velocities[-1, :] = np.random.normal(size=(2))
                self.velocities = new_velocities

            else:
                # deletion
                index = self.peers.index(diff)
                # https://stackoverflow.com/questions/19286657/index-all-except-one-item-in-python
                self.positions = self.positions[np.arange(len(self.positions)) != index]
                self.velocities = self.velocities[
                    np.arange(len(self.velocities)) != index
                ]
                self.peers.remove(diff)
                # print(f"removed {diff.index}")
        # print(self.peers)
        # apply forces
        N = len(self.peers)
        if N == 0:
            return

        directions = np.zeros((N, N, 2))
        weights = np.zeros((N, N))

        total_vel = 0
        pull_const = N
        dt = 0.01
        min_dist = 0.4
        separation_force = 20
        for x in range(N):
            for y in range(N):
                if self.peers[x] in self.peers[y].peers:
                    weights[x, y] = 1
                else:
                    weights[x, y] = 5

        weights = np.expand_dims(weights, -1)

        for x in range(N):
            for y in range(N):
                directions[x, y] = self.positions[y] - self.positions[x]

        distance_matrix = cdist(self.positions, self.positions)
        distance_matrix[distance_matrix == 0] = 100
        distance_matrix = np.expand_dims(distance_matrix, -1)

        # push opposite particles away
        forces = weights / distance_matrix * directions
        # pull like particles together
        forces += distance_matrix / weights * -directions
        # push particles too close apart
        push_apart = np.zeros(distance_matrix.shape)
        push_apart[distance_matrix < min_dist] = (
            separation_force / distance_matrix[distance_matrix < min_dist]
        )
        forces += push_apart * directions

        total_forces = np.sum(forces, 0)

        total_forces -= self.positions * pull_const  # drag points closer to origin
        self.velocities *= 0.8  # damping self.velocities
        self.velocities += total_forces * dt

        # TODO TESTING REMOVE THIS
        # self.velocities = total_forces * dt

        self.positions += self.velocities * dt * 5

        xmin = np.min(self.positions[:, 0]) * 1.1
        xmax = np.max(self.positions[:, 0]) * 1.1
        ymin = np.min(self.positions[:, 1]) * 1.1
        ymax = np.max(self.positions[:, 1]) * 1.1

        # print(xmin)
        # print(self.camera_coords)

        xmin = np.interp([0.5], [0, 1], [self.camera_coords[0], xmin]).item()
        xmax = np.interp([0.5], [0, 1], [self.camera_coords[1], xmax]).item()
        ymin = np.interp([0.5], [0, 1], [self.camera_coords[2], ymin]).item()
        ymax = np.interp([0.5], [0, 1], [self.camera_coords[3], ymax]).item()

        self.camera_coords = [xmin, xmax, ymin, ymax]

    def render(self):
        N = len(self.peers)
        if N == 0:
            return
        size = 2048
        xmin, xmax, ymin, ymax = self.camera_coords

        width = xmax - xmin
        height = ymax - ymin
        if width > 0 and height > 0:
            img = np.zeros((size, size, 3))

            def point_coords(index):
                # print(f"pc {index}")
                point = self.positions[index]
                x = int(size * (point[0] - xmin) / width)
                y = int(size * (point[1] - ymin) / height)
                return x, y

            def draw_point(index, img):
                # print(self.peers[index].team)
                # print(self.count)
                # print(
                #     np.array(
                #         hsv_to_rgb(self.peers[index].team * 1 / (self.count + 1), 1, 1)
                #     )
                # )
                return cv2.circle(
                    img,
                    point_coords(index),
                    6,
                    np.array(
                        hsv_to_rgb(self.peers[index].team * 1 / (self.count + 1), 1, 1)
                    )
                    * 255,
                    thickness=-1,
                )

            for i in range(N):
                for peer in self.peers[i].peers:
                    assert (
                        peer in self.peers
                    ), f"{self.peers} : {self.peers[i]} : {peer}"
                    cv2.line(
                        img,
                        point_coords(i),
                        point_coords(self.peers.index(peer)),
                        [255, 0, 0],
                        2,
                    )
            for i in range(N):
                img = draw_point(i, img)
            return img

    def show_img(self):
        img = self.render()
        cv2.imshow("step", img)
        key = cv2.waitKey(50)
        # print(key)
        if key == ord(" "):
            print("pause")
            while True:
                key = cv2.waitKey(1)
                if key == ord(" "):
                    break
        if key == ord("q"):
            assert False, "quit simulation"

    def connect_subsets(self, peers):
        print("connect subsets")
        self.step(peers)
        N = len(self.peers)
        # print(N)

        for i in range(N):
            self.peers[i].team = -1

        self.count = 0
        for i in range(N):
            if self.peers[i].team == -1:
                self.peers[i].team = self.count
                queue = self.peers[i].peers[:]
                closed = set([self.peers[i]])
                while len(queue):
                    v = queue.pop(0)
                    v.team = self.count
                    closed.add(v)
                    queue.extend(
                        list(
                            filter(
                                lambda x: x not in queue and x not in closed, v.peers[:]
                            )
                        )
                    )
                self.count += 1

        return self.count
