#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int is_digit(char ch) {
    return (ch >= '0' && ch <= '9');
}

int is_valid_int(char *string) {
    int str_len = strlen(string);
    if (str_len == 1) {
        return is_digit(string[0]);
    }
    else if (is_digit(string[0]) || string[0] == '+') {
        int i;
        for (i = 1; i < str_len; i++) {
            if (!is_digit(string[i])) {
                return 0;
            }
        }
        return 1;
    }
    return 0;
}

int is_zero(char *num) {
    int str_len = strlen(num);
    return ((str_len == 1 && num[0] == '0') || (str_len == 2 && num[0] == '+' && num[1] == '0'));
}

int is_one(char *num) {
    int str_len = strlen(num);
    return ((str_len == 1 && num[0] == '1') || (str_len == 2 && num[0] == '+' && num[1] == '1'));
}

int is_valid_text(char *name) {
    int str_len = strlen(name);
    return str_len > 4 && name[str_len-4] == '.' && name[str_len-3] == 't' && name[str_len-2] == 'x' && name[str_len-1] == 't';
}

int tremination(int error) {
    if (error == 0) {
        printf("Invalid Input!");
    }
    else {
        printf("An Error Has Occurred");
    }
    exit(1);
    return 1;
}

int is_valid(int argc, char *args[]) {
    if (argc > 4 || argc < 3) {
        tremination(0);
    }
    if (!is_valid_int(args[0]) || is_zero(args[0]) || is_one(args[0])) {
        tremination(0);
    }
    else if (argc == 3) {
        if (!is_valid_text(args[1]) || !is_valid_text(args[2])) {
            tremination(0);
        }
    }
    else {
        if (!is_valid_int(args[1]) || is_zero(args[1]) || !is_valid_text(args[2]) || !is_valid_text(args[3])) {
            tremination(0);
        }
    }

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

int convergence(double *prev[], double *curr[], int k, int d) {
    double epsilon = 0.001;
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

int kmeans(int k, int max_iter, char *input_filename, char *output_filename) {
    int d = 1;
    int n = 1;
    int i,j;
    int iter_cnt = 0;
    int stop = 0;
    char c;
    double *space_data_points;
    double **data_points;
    double *space_prev_centroids;
    double **prev_centroids;
    double *space_curr_centroids;
    double **curr_centroids;

    FILE *inp, *out;
    inp = fopen(input_filename, "r");
    if (inp == NULL) {
        fclose(inp);
        tremination(1);
    }
    while ((c = fgetc(inp)) != '\n') {
        if (c == ',') {
            d += 1;
        }
    }
    while ((c = fgetc(inp)) != EOF) {
        if (c == '\n') {
            n++;
        }
    }
    if (k >= n) {
        fclose(inp);
        tremination(0);
    }

    space_data_points = calloc(n*d, sizeof(double)); 
    data_points = calloc(n, sizeof(double*));
    if (space_data_points == NULL || data_points == NULL) {
        fclose(inp);
        free(space_data_points);
        free(data_points);
        tremination(1);
    }
    for (i = 0; i < n; i++) {
        data_points[i] = space_data_points + i*d;
    } 

    rewind(inp);
    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            fscanf(inp, "%lf,", &(data_points[i][j]));
        }
    }

    space_prev_centroids = calloc(k*d, sizeof(double)); 
    prev_centroids = calloc(k, sizeof(double*));
    if (space_prev_centroids == NULL || prev_centroids == NULL) {
        fclose(inp);
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
        fclose(inp);
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
        curr_centroids[i][j] = data_points[i][j];
        prev_centroids[i][j] = data_points[i][j];
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
            fclose(inp);
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
                    fclose(inp);
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

        if (convergence(prev_centroids, curr_centroids, k, d)) {
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
    
    out = fopen(output_filename, "w");
    if (out == NULL) {
        fclose(inp);
        fclose(out);
        free(space_data_points);
        free(data_points);
        free(space_prev_centroids);
        free(prev_centroids);
        free(space_curr_centroids);
        free(curr_centroids);
        tremination(1);
    }
    for (i = 0; i < k; i++) {
        for (j = 0; j <d; j++) {
            if (j < d-1) {
                fprintf(out, "%.4f,", curr_centroids[i][j]);
            }
            else {
                fprintf(out, "%.4f\n", curr_centroids[i][j]);
            }
        }
    }

    fclose(inp);
    fclose(out);
    free(space_data_points);
    free(data_points);
    free(space_prev_centroids);
    free(prev_centroids);
    free(space_curr_centroids);
    free(curr_centroids);

    return 0;
}

int main(int argc, char* argv[]) {
    int k;
    is_valid(argc-1, ++argv);
    sscanf(argv[0], "%d", &k);
    if (argc-1 == 3) {
        kmeans(k, 200, argv[1], argv[2]);
    }
    else {
        int max_iter;
        sscanf(argv[1], "%d", &max_iter);
        kmeans(k, max_iter, argv[2], argv[3]);
    }
    return 0;
}
