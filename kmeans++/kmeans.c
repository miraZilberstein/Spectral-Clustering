#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Python.h>

double *space_data_points;
double **data_points;
double *space_prev_centroids;
double **prev_centroids;
double *space_curr_centroids;
double **curr_centroids;
PyObject *final_list;

int tremination(int error) {
    printf("An Error Has Occurred");
    exit(1);
    return 1;
}

double euclidean_distance(double *v1, double *v2, int d) {
    double curr_sum = 0;
    int i;
    for (i = 0; i < d; i++) {
        curr_sum += pow((v1[i] - v2[i]),2);
    }
    return curr_sum;
}

double euclidean_norm(double *vector, int d) {
    double norm = 0;
    int i;
    for (i = 0; i < d; i++) {
        norm += pow(vector[i],2);
    }
    return pow(norm,0.5);
}

int convergence(double *prev[], double *curr[], int k, int d, double epsilon) {
    int i;
    for (i = 0; i < k; i++) {
        double prev_norm = euclidean_norm(prev[i], d);
        double curr_norm = euclidean_norm(curr[i], d);
        double cent_dist = fabs(curr_norm - prev_norm);
        if (cent_dist >= epsilon) {
            return 0;
        }
    }
    return 1;
}

PyObject *kmeans(int d, int k, int n, int max_iter, double epsilon, PyObject *data_points_py, PyObject *centroids_py) {
    int i,j;
    int iter_cnt = 0;
    int stop = 0;
    PyObject *tmp_list;

    space_data_points = calloc(n*d, sizeof(double)); 
    data_points = calloc(n, sizeof(double*));
    if (space_data_points == NULL || data_points == NULL) {
        free(space_data_points);
        free(data_points);
        tremination(1);
    }
    for (i = 0; i < n; i++) {
        data_points[i] = space_data_points + i*d;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            data_points[i][j] = PyFloat_AsDouble(PyList_GetItem(data_points_py, i*d + j));
        }
    }

    space_prev_centroids = calloc(k*d, sizeof(double)); 
    prev_centroids = calloc(k, sizeof(double*));
    if (space_prev_centroids == NULL || prev_centroids == NULL) {
        free(space_data_points);
        free(data_points);
        free(space_prev_centroids);
        free(prev_centroids);
        tremination(1);
    }
    for (i = 0; i < k; i++) {
        prev_centroids[i] = space_prev_centroids + i*d;
    }

    space_curr_centroids = calloc(k*d, sizeof(double)); 
    curr_centroids = calloc(k, sizeof(double*));
    if (space_curr_centroids == NULL || curr_centroids == NULL) {
        free(space_data_points);
        free(data_points);
        free(space_prev_centroids);
        free(prev_centroids);
        free(space_curr_centroids);
        free(curr_centroids);
        tremination(1);
    }
    for (i = 0; i < k; i++) {
        curr_centroids[i] = space_curr_centroids + i*d;
    }

    for (i = 0; i < k; i++) {
        for (j = 0; j < d; j++) {
            curr_centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(centroids_py, i*d + j));
            prev_centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(centroids_py, i*d + j));
        }
    }

    while (!stop && iter_cnt < max_iter) {
        int *size_cluster;
        double *space_clusters_sum; 
        double **clusters_sum;
        double min_dist;
        double dist;
        size_cluster = calloc(k, sizeof(int));
        space_clusters_sum = calloc(k*d, sizeof(double)); 
        clusters_sum = calloc(k, sizeof(double*));
        if (space_clusters_sum == NULL || clusters_sum == NULL || size_cluster == NULL) {
            free(space_data_points);
            free(data_points);
            free(space_prev_centroids);
            free(prev_centroids);
            free(space_curr_centroids);
            free(curr_centroids);
            free(size_cluster);
            free(space_clusters_sum);
            free(clusters_sum);
            tremination(1);
        }
        for (i = 0; i < k; i++) {
            clusters_sum[i] = space_clusters_sum + i*d;
        }

        for (i = 0 ; i < n; i++) {
            int ret_j = 0;
            min_dist = euclidean_distance(data_points[i], curr_centroids[0], d);
            for (j = 0; j < k; j++) {
                dist = euclidean_distance(data_points[i], curr_centroids[j], d);
                if (dist < min_dist) {
                    min_dist = dist;
                    ret_j = j;
                }
            }
            size_cluster[ret_j]++;
            for (j = 0; j < d; j++) {
                clusters_sum[ret_j][j] += data_points[i][j];
            }
        }
        for (i = 0; i < k; i++) {
            for (j = 0; j < d; j++) {
                if (size_cluster == 0) {
                    free(space_data_points);
                    free(data_points);
                    free(space_prev_centroids);
                    free(prev_centroids);
                    free(space_curr_centroids);
                    free(curr_centroids);
                    free(size_cluster);
                    free(space_clusters_sum);
                    free(clusters_sum);
                    tremination(1);
                }
                curr_centroids[i][j] = clusters_sum[i][j] / size_cluster[i]; 
            }
        }

        if (convergence(prev_centroids, curr_centroids, k, d, epsilon)) {
            stop = 1;
        }
        iter_cnt++;

        for (i = 0; i < k; i++) {
            for(j = 0; j < d; j++) {
                prev_centroids[i][j] = curr_centroids[i][j];
            }
        }
        free(size_cluster);
        free(space_clusters_sum);
        free(clusters_sum);
    }

    tmp_list = PyList_New(d*k);
    for (i = 0; i < k; i++) {
        for (j = 0; j <d; j++) {
            PyList_SetItem(tmp_list, i*d + j,PyFloat_FromDouble(curr_centroids[i][j]));
        }
    }

    free(space_data_points);
    free(data_points);
    free(space_prev_centroids);
    free(prev_centroids);
    free(space_curr_centroids);
    free(curr_centroids);

    return tmp_list;
}

static PyObject *fit(PyObject *self, PyObject *args) {
    int d_py = 0;
    int k_py = 0;
    int n_py = 0;
    int max_iter_py = 0;
    double epsilon_py = 0.0;
    PyObject *data_points_py = NULL;
    PyObject *centroids_py = NULL;
    if (!PyArg_ParseTuple(args, "iiiidOO", &d_py, &k_py, &n_py, &max_iter_py, &epsilon_py, &data_points_py, &centroids_py)) {
        return NULL;
    }

    PyObject *final_list = PyList_New(d_py*k_py);
    final_list = kmeans(d_py, k_py, n_py, max_iter_py, epsilon_py, data_points_py, centroids_py);
    return final_list;
}

static PyMethodDef myMethods[] = {
    {"fit", fit, METH_VARARGS, "K-Means Clustering Algorithm."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mykmeanssp = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    "K-Means Module",
    -1,
    myMethods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&mykmeanssp);
}
