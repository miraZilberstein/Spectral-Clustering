import sys
import mykmeanssp
import pandas as pd, numpy as np

def is_valid_int(string):
    if len(string) == 1:
        return string[0].isdigit()
    elif string[0].isdigit() or string[0] == "+":
        return string[1:].isdigit()
    else:
        return False

def is_valid_eps(num):
    try:
        float(num)
        if float(num) < 0:
            return False
        return True
    except ValueError:
        return False

def is_zero(num):
    return num == "0" or num == "+0"

def is_one(num):
    return num == "1" or num == "+1"

def is_valid_file(name):
    return (len(name) > 4 and name[-4:] == ".txt") or (len(name) > 4 and name[-4:] == ".csv")

def termination(error):
    if error == 0:
        print("Invalid Input!")
    else:
        print("An Error Has Occurred")
    sys.exit()

def is_valid(args):
    if len(args) > 5 or len(args) < 4: # too much or not enough args
        termination(0)

    if not is_valid_int(args[0]) or is_zero(args[0]) or is_one(args[0]): # k <= 1
        termination(0)

    elif len(args) == 4:
        if (not is_valid_file(args[2])) or (not is_valid_file(args[3])): # input files are not finished with .txt or .csv
            termination(0)
        if not is_valid_eps(args[1]): # epsilon is not float/ epsilon is negative
            termination(0)

    else:  # len(args) == 5
        if (not is_valid_int(args[1])) or is_zero(args[1]):  # max_iter isn't a positive number
            termination(0)
        if (not is_valid_file(args[3])) or (not is_valid_file(args[4])): # input files are not finished with .txt or .csv
            termination(0)
        if not is_valid_eps(args[2]): # epsilon is not float/epsilon is negative
            termination(0)

    return True

def d_l(x_l, centroid_j):
    s = np.subtract(x_l, centroid_j)
    return np.dot(s, s)

def kmeans_pp(k, n, data):
    i = 0
    np.random.seed(0)
    centroids_index = np.array([np.random.choice(n)])
    while(i < k-1):
        D = np.array(range(n),dtype = float)
        P = np.array(range(n),dtype = float)
        for l in range(n):
            x_l = np.array(data[l, 1:])
            D[l] = np.min(np.sum(np.square(np.subtract(x_l,data[centroids_index[:],1:])),axis=1))
        Dm = np.sum(D)
        P = np.divide(D, Dm)
        i+=1
        centroids_index = np.append(centroids_index, [np.random.choice(n, p=P)])   

    return centroids_index

def print_solution(vals, indices, d, k):
    indices_str = ""
    for index in indices:
        indices_str += str(index)
        indices_str += ","
    print(indices_str[:-1])

    for i in range(k):
        ctrd = ""
        for j in range(d):
            ctrd += "{:.4f}".format(vals[i*d + j])
            ctrd += ","
        print(ctrd[:-1])    

def main():
    args = sys.argv[1:]
    is_valid(args)
    if len(args) == 4:
        f1 = pd.read_csv(args[2],header=None)
        f2 = pd.read_csv(args[3],header=None)

    else:  # len(args) != 4
        f1 = pd.read_csv(args[3],header=None)
        f2 = pd.read_csv(args[4],header=None)

    f_join = pd.merge(f1, f2, on=0, how='inner')
    f_join.sort_values(by=[0], inplace=True)
    d = len(f_join.columns) - 1
    n = len(f_join)
    data = f_join.to_numpy()
    k = int(args[0])

    if k >= n:
        termination(0)

    if len(args) == 4:
        epsilon = float(args[1])
        max_iter = 300
        centroids_index = kmeans_pp(k, n, data)
    else:
        epsilon = float(args[2])
        max_iter = int(args[1])
        centroids_index = kmeans_pp(k, n, data)

    data_points = list(np.concatenate(data[:, 1:]))
    centroids = list(np.concatenate(data[centroids_index, 1:]))
    res = mykmeanssp.fit(d, k, n, max_iter, epsilon, data_points, centroids)
    if res == None:
        termination(1)
    print_solution(res, centroids_index, d, k)

if __name__ == "__main__":
    main()
