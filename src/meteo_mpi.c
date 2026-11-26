/*
 * MeteoPel - Versão Distribuída em MPI (com suporte híbrido opcional OpenMP)
 *
 * Estratégia distribuída:
 *   1. Rank 0 carrega/expande as amostras e constrói/carrega a RandomForest (30.400 B).
 *   2. MPI_Bcast replica N e a RandomForest para todos os processos.
 *   3. MPI_Scatterv distribui fatias balanceadas das amostras (suporta N % P != 0).
 *   4. Cada rank computa sua partição local (em paralelo com OpenMP se habilitado).
 *   5. MPI_Gatherv consolida as previsões no Rank 0.
 *
 * Compilação: mpicc -O3 -fopenmp -o meteo_mpi src/meteo_mpi.c -lm
 * Execução:   mpirun -np 4 ./meteo_mpi dados_pelotas.csv
 */

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../include/forest_common.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    CliOptions opt;
    if (cli_parse(argc, argv, &opt)) {
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    int num_samples = 0;
    int status_ok = 1;
    float* all_features = NULL;
    float* all_predictions = NULL;
    RandomForest* forest = (RandomForest*)malloc(sizeof(RandomForest));
    float min_values[NUM_FEATURES] = {0};
    float max_values[NUM_FEATURES] = {0};

    if (!forest) {
        fprintf(stderr, "[Rank %d] Erro ao alocar RandomForest.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        printf("=== RANDOM FOREST - INFERENCIA MPI ===\n\n");
        printf("Arquivo: %s\n", opt.csv_filename);

        int base_samples = 0;
        float* base_features = read_csv_data(opt.csv_filename, &base_samples);
        if (!base_features) {
            status_ok = 0;
        } else {
            num_samples = base_samples;
            all_features = base_features;
            if (opt.expand_n > 0) {
                all_features = expand_features_cyclic(base_features, base_samples, opt.expand_n);
                free(base_features);
                if (!all_features) {
                    status_ok = 0;
                } else {
                    num_samples = opt.expand_n;
                }
            }
        }

        if (status_ok) {
            printf("Amostras carregadas: %d\n", num_samples);
            compute_feature_ranges_seq(all_features, num_samples, min_values, max_values);
            print_feature_ranges(min_values, max_values);

            if (opt.model_filename) {
                if (!load_forest_bin(opt.model_filename, forest)) {
                    status_ok = 0;
                } else {
                    printf("\nModelo carregado: %s\n", opt.model_filename);
                }
            } else if (opt.legacy_rand) {
                build_random_forest_legacy(forest, min_values, max_values);
                printf("\nModelo: floresta sintetica (rand legado, seed=%u)\n", FOREST_SEED_DEFAULT);
            } else {
                build_random_forest_stateless_seq(forest, min_values, max_values);
                printf("\nModelo: floresta sintetica deterministica (seed=%u)\n", FOREST_SEED_DEFAULT);
            }

            if (status_ok) {
                all_predictions = (float*)malloc((size_t)num_samples * sizeof(float));
                if (!all_predictions) status_ok = 0;
            }
        }
    }

    MPI_Bcast(&status_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!status_ok) {
        free(all_features);
        free(all_predictions);
        free(forest);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Bcast(&num_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int* sample_counts = (int*)malloc((size_t)world_size * sizeof(int));
    int* sample_displs = (int*)malloc((size_t)world_size * sizeof(int));
    int* feat_counts = (int*)malloc((size_t)world_size * sizeof(int));
    int* feat_displs = (int*)malloc((size_t)world_size * sizeof(int));

    int base_chunk = num_samples / world_size;
    int remainder = num_samples % world_size;
    int offset = 0;
    for (int p = 0; p < world_size; p++) {
        sample_counts[p] = base_chunk + (p < remainder ? 1 : 0);
        sample_displs[p] = offset;
        feat_counts[p] = sample_counts[p] * NUM_FEATURES;
        feat_displs[p] = offset * NUM_FEATURES;
        offset += sample_counts[p];
    }

    int local_n = sample_counts[rank];
    float* local_features = (float*)malloc((size_t)(local_n > 0 ? local_n : 1) * NUM_FEATURES * sizeof(float));
    float* local_preds = (float*)malloc((size_t)(local_n > 0 ? local_n : 1) * sizeof(float));
    double* total_times = (double*)malloc((size_t)opt.repeat * sizeof(double));
    double* compute_times = (double*)malloc((size_t)opt.repeat * sizeof(double));

    int total_runs = opt.warmup + opt.repeat;
    for (int run = 0; run < total_runs; run++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t_total_ini = MPI_Wtime();

        MPI_Bcast(forest, (int)sizeof(RandomForest), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(all_features, feat_counts, feat_displs, MPI_FLOAT,
                     local_features, local_n * NUM_FEATURES, MPI_FLOAT,
                     0, MPI_COMM_WORLD);

        double t_comp_ini = MPI_Wtime();
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < local_n; i++) {
            local_preds[i] = predict_sample_cpu(&local_features[(size_t)i * NUM_FEATURES], forest);
        }
        double local_comp = MPI_Wtime() - t_comp_ini;

        MPI_Gatherv(local_preds, local_n, MPI_FLOAT,
                    all_predictions, sample_counts, sample_displs, MPI_FLOAT,
                    0, MPI_COMM_WORLD);

        double local_total = MPI_Wtime() - t_total_ini;
        double max_comp = 0.0, max_total = 0.0;
        MPI_Reduce(&local_comp, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0 && run >= opt.warmup) {
            int idx = run - opt.warmup;
            compute_times[idx] = max_comp;
            total_times[idx] = max_total;
        }
    }

    if (rank == 0) {
        double med_comp = median_double(compute_times, opt.repeat);
        double med_total = median_double(total_times, opt.repeat);
        #ifdef _OPENMP
        int omp_t = omp_get_max_threads();
        #else
        int omp_t = 1;
        #endif
        printf("\nProcessos MPI: %d (OpenMP por rank: %d)\n", world_size, omp_t);
        printf("Tempo de inferencia (computacao local max): %.6f segundos\n", med_comp);
        printf("Tempo de inferencia (total com Bcast+Scatterv+Gatherv): %.6f segundos\n", med_total);

        if (opt.dump_filename) {
            dump_predictions_bin(opt.dump_filename, all_predictions, num_samples);
        }

        print_rain_forecast(all_predictions, num_samples, opt.forecast_tail);
    }

    free(sample_counts);
    free(sample_displs);
    free(feat_counts);
    free(feat_displs);
    free(local_features);
    free(local_preds);
    free(total_times);
    free(compute_times);
    free(all_features);
    free(all_predictions);
    free(forest);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
