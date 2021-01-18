import numpy as np
import time
from tqdm import tqdm
import pandas as pd
from drawer import NetworkDrawer
import cv2

# want to test this hypothesis:
# randomly selecting peers is inviable for producing the lowest average + max path length between any two nodes

MAX_PEERS = 8


class Peer(object):
    def __init__(self, index, ttl, bandwidth=8):
        self.index = index
        self.peers = []
        self.ttl = int(ttl)
        self.alive = 0
        self.bandwidth = max(4, int(bandwidth))
        self.team = -1

    def random_connect(self, peers_list):
        if len(self.peers) == 0:
            # connect first to a random peer
            free_peers = list(
                filter(
                    lambda x: x not in self.peers
                    and len(x.peers) < x.bandwidth
                    and x.index != self.index,
                    peers_list,
                )
            )
            if len(free_peers):
                i = np.random.randint(len(free_peers))
                # print("rc connect")
                self.connect(free_peers[i])

        if len(self.peers) > 0 and len(self.peers) < self.bandwidth:
            # now bfs through peer's peers and add more connections
            queue = self.peers[:]
            closed = set(self.peers + [self])
            while len(queue):
                v = queue.pop(0)
                if len(v.peers) < v.bandwidth and v not in closed:
                    # found a free peer!
                    # print("bfs connect")
                    self.connect(v)
                    if len(self.peers) == self.bandwidth:
                        break
                closed.add(v)
                queue.extend(
                    list(
                        filter(
                            lambda peer: peer not in closed and peer not in queue,
                            v.peers,
                        )
                    )
                )
        for peer in self.peers:
            assert self in peer.peers

    def connect(self, peer):
        assert len(self.peers) < self.bandwidth
        assert len(peer.peers) < peer.bandwidth
        assert peer not in self.peers and self not in peer.peers
        peer.peers.append(self)
        self.peers.append(peer)

    def disconnect(self, peer):
        assert peer in self.peers and self in peer.peers
        peer.peers.remove(self)
        self.peers.remove(peer)

    def step(self, peers_list):
        self.alive += 1
        for peer in self.peers:
            assert self in peer.peers, f"{self} {peer}"

        if self.alive > self.ttl:
            # disconnect from the network
            for peer in self.peers[:]:
                self.disconnect(peer)
            peers_list.remove(self)
            return
        if len(self.peers) < self.bandwidth:
            self.random_connect(peers_list)

    def bfs(self, target):
        queue = self.peers[:]
        closed = set()
        came_from = {}
        while len(queue):
            v = queue.pop(0)
            if v.index == target.index:
                # found target
                path = [v]
                while v in came_from:
                    v = came_from[v]
                    path.append(v)
                return path
            else:
                closed.add(v)
                for peer in filter(
                    lambda peer: peer not in closed and peer not in queue, v.peers
                ):
                    queue.append(peer)
                    came_from[peer] = v

    def __repr__(self):
        return f"<Peer {self.index} {self.ttl} {list(map(lambda x : x.index, self.peers))}>"


def simulate(n=10000, ttl_mu=20, ttl_sigma=5, neighbours_mu=8, draw=False):
    network = []
    drawer = NetworkDrawer()
    for i in tqdm(range(n)):
        # print(f"iter {i}")
        # print(network)
        for peer in network:
            peer.step(network)
        # print(network)
        for peer in network:
            for other in peer.peers:
                assert other in network

        if len(network) < 1000:
            new_peer = Peer(
                i,
                np.random.normal(ttl_mu, ttl_sigma),
                np.random.normal(neighbours_mu, 2),
            )
            network.append(new_peer)
            new_peer.random_connect(network)
            if len(new_peer.peers) == 0 and i > 0:
                # new peer couldn't connect, remove them
                network.remove(new_peer)

        # if draw and i % 1000 == 0:
        #     for i in range(1):
        #         drawer.step(network)

    total_dist = 0
    max_dist, min_dist = 0, 0
    total_pairs = 0
    no_paths = 0
    # print(network)
    for a in network:
        for b in network:
            if b.index > a.index:
                total_pairs += 1
                path = a.bfs(b)
                if path is not None:
                    total_dist += len(path)
                    max_dist = max(max_dist, len(path))
                    min_dist = min(min_dist, len(path))
                else:
                    no_paths += 1

    return network, total_pairs, total_dist, max_dist, min_dist, no_paths


data = {
    "ttl_mu": [],
    "nmax_mu": [],
    "network": [],
    "pairs": [],
    "dist": [],
    "max_dist": [],
    "min_dist": [],
    "no_paths": [],
    "subsets": [],
}

for ttl_mu in [20, 50, 100, 200]:
    for neighbours_mu in [4, 6, 8, 10, 12]:
        network, pairs, dist, max_dist, min_dist, no_paths = simulate(
            2000, ttl_mu=ttl_mu, neighbours_mu=neighbours_mu, draw=True
        )

        data["ttl_mu"].append(ttl_mu)
        data["nmax_mu"].append(neighbours_mu)
        data["network"].append(len(network))
        data["pairs"].append(pairs)
        data["dist"].append(dist)
        data["max_dist"].append(max_dist)
        data["min_dist"].append(min_dist)
        data["no_paths"].append(no_paths)
        drawer = NetworkDrawer()
        subsets = drawer.connect_subsets(network)
        data["subsets"].append(subsets)
        df = pd.DataFrame.from_dict(data)
        print(df)
        print(df[["ttl_mu", "nmax_mu", "dist", "max_dist", "subsets"]].to_latex())

        for i in tqdm(range(1000)):
            drawer.step(network)

        img = drawer.render()
        cv2.imwrite(f"./images/{ttl_mu}_{neighbours_mu}_{subsets}.png", img)
        # while True:
        #     try:
        #         for i in range(30):
        #             drawer.step(network)
        #         drawer.render()
        #     except:
        #         print("drawer break")
        #         break

print(pd.DataFrame.from_dict(data))
