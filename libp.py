import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import statistics
import random
import math

# Step 1, get the data, json formatted as an array of [X,Y] arrays
def load_data(filename):
    fp = open(filename)
    data = json.load(fp)
    fp.close()
    X, Y = [item[0] for item in data], [item[1] for item in data]
    plt.scatter(np.array(X), np.array(Y))
    plt.show()
    plt.clf()
    return X, Y


#####################################################################

# Step 1.5, partition the space, and return the list of representative points along with the respective densities.
class Node:
    def __init__(self, X, Y, threshold):
        self.X = X
        self.Y = Y
        self.left = object
        self.right = object
        self.leaf = False
        self.split(threshold)

    def split(self, threshold):
        if len(self.X) <= threshold:
            self.leaf = True
            return
        diff_x, diff_y = max(self.X) - min(self.X), max(self.Y) - min(self.Y)
        x_left = []
        x_right = []
        y_left = []
        y_right = []
        med_x = statistics.median(self.X)
        med_y = statistics.median(self.Y)
        if diff_x > diff_y:
            x_left = [self.X[i] for i in range(len(self.X)) if self.X[i] < med_x]
            y_left = [self.Y[i] for i in range(len(self.Y)) if self.X[i] < med_x]
            x_right = [self.X[i] for i in range(len(self.X)) if self.X[i] > med_x]
            y_right = [self.Y[i] for i in range(len(self.Y)) if self.X[i] > med_x]
        else:
            x_left = [self.X[i] for i in range(len(self.X)) if self.Y[i] < med_y]
            y_left = [self.Y[i] for i in range(len(self.Y)) if self.Y[i] < med_y]
            x_right = [self.X[i] for i in range(len(self.X)) if self.Y[i] > med_y]
            y_right = [self.Y[i] for i in range(len(self.Y)) if self.Y[i] > med_y]

        if len(x_left) > len(x_right):
            x_right.append(med_x)
            y_right.append(med_y)
        else:
            x_left.append(med_x)
            y_left.append(med_y)

        self.left = Node(x_left, y_left, threshold)
        self.right = Node(x_right, y_right, threshold)

    def buckets(self):
        if self.leaf:
            return [[self.X, self.Y]]
        else:
            return self.left.buckets() + self.right.buckets()


#Step 2: Get the buckets, and plot them
def get_buckets(X, Y, threshold):
    root = Node(X, Y, threshold)
    buckets = root.buckets()
    random.shuffle(buckets)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(buckets))))
    for bucket in buckets:
        plt.scatter(np.array(bucket[0]), np.array(bucket[1]), color=next(colors))
    plt.show()
    plt.clf()
    return buckets


######################################################################################


#Step 3: reduce the buckets, and return a list of the reduced po1ints, along with normalized density
def normalize_array(A):
    return [x / min(A) for x in A]


def get_densities(buckets):
    D = [len(X) / float((max(X) - min(X)) * (max(Y) - min(Y)) + 1) for [X, Y] in buckets]
    return normalize_array(D)


def get_representatives(buckets):
    x_reps = [statistics.mean(bucket[0]) for bucket in buckets]
    y_reps = [statistics.mean(bucket[1]) for bucket in buckets]
    return x_reps, y_reps


def reduce_buckets(buckets):
    densities = get_densities(buckets)
    x_reps, y_reps = get_representatives(buckets)
    return x_reps, y_reps, densities


############################################################################################

#Step 4: Generate a complete graph, find the MST, and keep removing edges till we hit gold
def make_cgraph(x_reps, y_reps, densities):
    complete_graph = np.zeros((len(x_reps), len(y_reps)))
    for i in range(len(x_reps)):
        for j in range(len(y_reps)):
            complete_graph[i][j] = math.sqrt(
                (x_reps[i] - x_reps[j]) ** 2
                + (y_reps[i] - y_reps[j]) ** 2
            )* math.log((densities[i] + densities[j]), 2)
    return complete_graph


def find_connected(cgraph, k, buckets):
    mst_csr = minimum_spanning_tree(cgraph)
    mst = mst_csr.toarray()
    (xnonzero, ynonzero) = mst_csr.nonzero()
    #Extract the nonsparse values and put them in a dict with their indices
    nzlist = [{'x': xnonzero[i], 'y': ynonzero[i], 'val': mst[xnonzero[i]][ynonzero[i]]} for i in range(len(xnonzero))]
    #Sort the list of dicts according to value
    nzsorted = sorted(nzlist, key=lambda k: k['val'])
    #cull the values not needed
    for e in nzsorted[1 - k:]:
        mst[e['x']][e['y']] = 0
    n, labels = connected_components(mst)
    newbuckets = [[[], []] for i in range(k)]
    for i in range(len(labels)):
        newbuckets[labels[i]][0] += buckets[i][0]
        newbuckets[labels[i]][1] += buckets[i][1]

    colors = iter(cm.rainbow(np.linspace(0, 1, len(newbuckets))))
    for bucket in newbuckets:
        plt.scatter(np.array(bucket[0]), np.array(bucket[1]), color=next(colors))
    plt.show()
    plt.clf()
    return newbuckets

def make_base_stations(newbuckets):
    bx = [sum(newbuckets[i][0])/len(newbuckets[i][0]) for i in range(len(newbuckets))]
    by = [sum(newbuckets[i][1])/len(newbuckets[i][1]) for i in range(len(newbuckets))]
    radii = [max(
                [math.sqrt((b[0][j] - bx[i])**2 + (b[1][j] - by[i])**2) for j in range(len(b[0]))]
                )
             for (i,b) in enumerate(newbuckets)]
    print(radii)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(newbuckets))))
    for idx,bucket in enumerate(newbuckets):
        #area = np.pi * radii[idx]**2
        c = next(colors)
        plt.scatter(np.array(bucket[0]), np.array(bucket[1]), color=c)
        #plt.scatter(bx[idx],by[idx],s=area,color=c,alpha=0.3)
        circle = plt.Circle((bx[idx],by[idx]),radii[idx],color=c,fill=True,alpha=0.2)
        plt.scatter(bx[idx],by[idx],color='black')
        plt.gca().add_patch(circle)
    plt.axis('scaled')
    plt.show()
    plt.clf()
    return bx, by, radii

def main(filename, k, threshold):
    x, y = load_data(filename)
    #threshold = int(len(x)/k)
    buckets = get_buckets(x, y, threshold)
    x_reps, y_reps, densities = reduce_buckets(buckets)
    newbuckets = find_connected(make_cgraph(x_reps,y_reps,densities),k,buckets)
    bx, by, radii = make_base_stations(newbuckets)
