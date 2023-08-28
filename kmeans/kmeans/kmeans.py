import sys

def is_valid_int(string):
    if len(string) == 1:
        return string[0].isdigit()
    elif string[0].isdigit() or string[0] == "+":
        return string[1:].isdigit()
    else:
        return False

def is_zero(num):
    return num == "0" or num == "+0"

def is_one(num):
    return num == "1" or num == "+1"

def is_valid_txt(name):
    return len(name) > 4 and name[-4:] == ".txt"

def termination(error):
    if error == 0:
        print("Invalid Input!")
    else:
        print("An Error Has Occurred")
    sys.exit()

def is_valid(args):
    if (len(args) > 4 or len(args) < 3): # too much or not enough args
        termination(0)
        
    if not is_valid_int(args[0]) or is_zero(args[0]) or is_one(args[0]): # k <= 1
        termination(0)

    elif (len(args) == 3):
        if (not is_valid_txt(args[1])) or (not is_valid_txt(args[2])): # input/output file isn't finished with .txt
            termination(0)

    else: # len(args) == 4
        if (not is_valid_int(args[1])) or is_zero(args[1]) or \
        (not is_valid_txt(args[2])) or (not is_valid_txt(args[3])): # max_iter isn't a positive number || input/output file isn't finished with .txt
            termination(0)

    return True 

def euclidean_distance(v1, v2, d):
    curr_sum = 0
    for i in range(d):
        curr_sum += ((v1[i] - v2[i])**2)
    return curr_sum # without sqrt

def euclidean_norm(vector, d):
    norm = 0
    for i in range(d):
        norm += (vector[i]**2)
    return norm**0.5

def convergence(prev, curr, k, d):
    epsilon = 0.001
    for i in range(k): # for each centroid
        prev_norm = euclidean_norm(prev[i], d)
        curr_norm = euclidean_norm(curr[i], d)
        cent_dist = abs(curr_norm - prev_norm)
        if cent_dist >= epsilon: # room for improvement
            return False
    return True

def kmeans(k, max_iter, input_filename, output_filename):
    inp = open(input_filename, 'r')
    data_points = inp.readlines()
    n = len(data_points)
    if k >= n: # k bigger-equal than n
        termination(0)

    try:
        for i in range(n):
            tmp_row = data_points[i].split(",")
            dp = []
            for elem in tmp_row:
                dp.append(float(elem))
            data_points[i] = dp

        prev_centroids = data_points[:k]
        curr_centroids = data_points[:k]
        d = len(data_points[0])
        iter_cnt = 0
        stop = False

        while (not stop) and (iter_cnt < max_iter):
            clusters = [[] for i in range(k)]
            for dp in data_points: # Assignment step
                min_dist = euclidean_distance(dp, curr_centroids[0], d)
                ret_j = 0
                for j in range(k):
                    dist = euclidean_distance(dp, curr_centroids[j], d)
                    if dist < min_dist:
                        min_dist = dist
                        ret_j = j
                clusters[ret_j].append(dp)

            for i in range(k): # Update step
                new_cntr = [0 for j in range(d)]
                for vector in clusters[i]:
                    for row in range(d):
                        new_cntr[row] += vector[row]
                num_of_vectors = len(clusters[i])
                for row in range(d):
                    new_cntr[row] /= num_of_vectors
                curr_centroids[i] = new_cntr

            if convergence(prev_centroids, curr_centroids, k, d): # Stop condition 1
                stop = True
            iter_cnt += 1 # Stop condition 2
            prev_centroids = curr_centroids.copy()

        out = open(output_filename, 'w')
        for vector in curr_centroids: # Output creation
            str_vec = ""
            for num in vector:
                final_num = "{:.4f}".format(num)
                str_vec += final_num + ","
            out.writelines(str_vec[:-1] + "\n")
        inp.close()
        out.close()
        return 0
 
    except:
        termination(1)
        return 1

def main():
    args = sys.argv[1:]
    is_valid(args)
    if len(args) == 3:
        kmeans(int(args[0]), 200, args[1], args[2])
    else:
        kmeans(int(args[0]), int(args[1]), args[2], args[3])

if __name__ == "__main__":
    main()
