import sys
import spkmeansmodule
import pandas as pd, numpy as np

def is_valid_int(string):
    if len(string) == 1:
        return string[0].isdigit()
    elif string[0].isdigit() or string[0] == "+":
        return string[1:].isdigit()
    else:
        return False

def is_valid_file(name):
    return (len(name) > 4 and name[-4:] == ".txt") or (len(name) > 4 and name[-4:] == ".csv")

def termination(error):
    if error == 0:
        print("Invalid Input!")
    else:
        print("An Error Has Occurred")
    sys.exit()

def is_valid(args):
    if len(args) != 3: # too much or not enough args
        termination(0)

    if not is_valid_int(args[0]):
        termination(0)

    if args[1] not in ("spk", "wam", "ddg", "lnorm", "jacobi"):
        termination(0)

    if not is_valid_file(args[2]): # input file is not finished with .txt or .csv
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
            x_l = np.array(data[l, 0:])
            D[l] = np.min(np.sum(np.square(np.subtract(x_l,data[centroids_index[:],0:])),axis=1))
        Dm = np.sum(D)
        P = np.divide(D, Dm)
        i+=1
        centroids_index = np.append(centroids_index, [np.random.choice(n, p=P)])   

    return centroids_index

def flat_to_matrix(flat, n, m):
    matrix = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(flat[i*m+j])
        matrix.append(row)
    return matrix

def eigengap_heuristic(eigen_vals):
    iterable = (abs(eigen_vals[i] - eigen_vals[i+1]) for i in (range(len(eigen_vals) // 2)))
    tmp = np.fromiter(iterable, dtype=float)
    res = tmp.argmax() + 1
    return res

def normalize(mat):
    ret_mat = []
    for i in range(len(mat)):
        row = mat[i]
        total = 0
        for val in row:
            total += (val**2)
        total = (total**0.5)
        if total == 0:
            ret_mat.append(row)
            continue
        for i in range(len(row)):
            row[i] /= total
        ret_mat.append(row)

    return np.array(ret_mat)

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
    try:
        args = sys.argv[1:]
    except:
        termination(0)
    
    is_valid(args)
    k = int(args[0])
    goal = args[1]
    filename_py = args[2]
    file_py = pd.read_csv(args[2],header=None)
    d = len(file_py.columns) - 1
    n = len(file_py)

    if k < 0 or k >= n:
        termination(0)

    if goal == "wam":
        spkmeansmodule.pyWam(filename_py)
        sys.exit()
    elif goal == "ddg":
        spkmeansmodule.pyDdg(filename_py)
        sys.exit()
    elif goal == "lnorm":
        spkmeansmodule.pyLnorm(filename_py)
        sys.exit()
    elif goal == "jacobi":
        spkmeansmodule.pyJacobi(filename_py)
        sys.exit()
    elif goal == "spk":
        spk_res = spkmeansmodule.pySpk(filename_py, n)
        spk_mat = np.array(flat_to_matrix(spk_res, n+1, n))

        eigen_vals = spk_mat[0,:]
        indices = np.argsort(-eigen_vals)

        if k == 0:
            k = eigengap_heuristic(eigen_vals[indices])

        # sort columns by decreasing order of eigenvalues (first k).
        U_mat = spk_mat[1:][:, indices[:k]]
        T_mat = normalize(U_mat)

        d = len(T_mat[0])
        centroids_index = kmeans_pp(k, n, T_mat)
        data_points = list(np.concatenate(T_mat))
        centroids = list(np.concatenate(T_mat[list(centroids_index)]))
        res = spkmeansmodule.fit(d, k, n, 300, 0, data_points, centroids)
        if res == None:
            termination(1)
        print_solution(res, centroids_index, d, k)

if __name__ == "__main__":
    main()
