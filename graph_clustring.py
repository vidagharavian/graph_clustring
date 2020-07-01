import numpy as np
from PIL import Image
from numpy import linalg as LA
from sklearn.cluster import KMeans


def read_image() -> list:
    im = Image.open('flower.jpg', 'r')
    return list(im.getdata())


def adjacency_matrix(data: list) -> list:
    A = []
    for element in data:
        a = []
        for element2 in data:
            e = np.exp(-sum((np.power((element[i] - element2[i]), 2)) for i in range(len(element2))))
            a.append(e)
        A.append(a)
    return A


def degree_matrix(A: list) -> list:
    count = 0
    D = []
    for element in A:
        row = []
        sum = 0
        for i in range(len(element)):
            row.append(0)
            sum += element[i] if i != count else 0

        row[count] = sum
        D.append(row)
        count += 1
    return D


def laplacian_matrix(D, A):
    L = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(D[i][j] - A[i][j])
        L.append(row)
    return L


def conductance(A):
    pass


def compute_lambda(L):
    w, v = LA.eig(np.array(L))
    m = np.argmax(w)
    v = v[:, np.argsort(w)]
    w = w[np.argsort(w)]
    return v.real


def ploting(v):
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(v)
    colors = kmeans.labels_
    return colors


def get_centers(colors: np.ndarray, data):
    centers = []
    clusters = []
    for i in range(colors.size):
        if not clusters.__contains__(colors[i]):
            clusters.append(colors[i])
            centers.append(data[i])

    return centers, clusters


def reshape_data(colors: np.ndarray, data):
    centers = []
    clusters = []
    for i in range(colors.size):
        if not clusters.__contains__(colors[i]):
            clusters.append(colors[i])
            centers.append(data[i])
        else:
            data[i] = centers[clusters.index(colors[i])]
    return data


def graph_clustering(data):
    A = adjacency_matrix(data)
    D = degree_matrix(A)
    L = laplacian_matrix(D, A)
    return compute_lambda(L)


def make_image(img, data):
    new_image = []
    for x in range(img.size[1]):
        new_image_row = []
        for i in range(img.size[0]):
            new_image_row.append(data[x * img.size[0] + i])
        new_image.append(new_image_row)
    new_image = np.asarray(new_image, dtype=np.uint8)
    new_image = Image.fromarray(new_image, 'RGB')
    new_image.show()
    new_image.save('flower_out2.jpg')


img = Image.open('flower.jpg', 'r')
data: list = read_image()
print(len(data))
# v = graph_clustering(data[:10])
colors: np.ndarray = ploting(data)
print(colors)
centers, cluster = get_centers(colors, data)
make_image(img=img, data=data)
counter = 0
colors = []
while counter < int(len(data) / 10):
    new_data = []
    new_data.append(data[counter * 10:(counter + 1) * 10])
    new_data = np.append(new_data, centers).reshape(18, 3).tolist()
    v = graph_clustering(new_data)
    new_colors = ploting(v)
    colors = np.append(colors, new_colors[:10])
    counter += 1
print(colors.size)
data =reshape_data(colors, data)
make_image(img=img, data=data)
