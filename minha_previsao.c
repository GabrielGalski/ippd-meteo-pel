/*
 * MeteoPel - Versão Paralela em Memória Compartilhada (OpenMP)
 * Paraleliza: (1) cálculo de faixas (reduction min/max),
 *             (2) construção da floresta sintética (PRNG stateless thread-safe),
 *             (3) inferência das amostras.
 *
 * Compilação direta (retrocompatível com README original):
 *   gcc -O3 -fopenmp -o meteo_omp minha_previsao.c -lm
 */

#include <omp.h>
#include "include/forest_common.h"

void compute_feature_ranges_omp(
    const float* features,
    int num_samples,
    float min_values[NUM_FEATURES],
    float max_values[NUM_FEATURES])
{
    float min0 = FLT_MAX, min1 = FLT_MAX, min2 = FLT_MAX, min3 = FLT_MAX, min4 = FLT_MAX;
    float max0 = -FLT_MAX, max1 = -FLT_MAX, max2 = -FLT_MAX, max3 = -FLT_MAX, max4 = -FLT_MAX;

    #pragma omp parallel for schedule(static) \
        reduction(min:min0, min1, min2, min3, min4) \
        reduction(max:max0, max1, max2, max3, max4)
    for (int i = 0; i < num_samples; i++) {
        const float* s = &features[(size_t)i * NUM_FEATURES];
        if (s[0] < min0) min0 = s[0];
        if (s[1] < min1) min1 = s[1];
        if (s[2] < min2) min2 = s[2];
        if (s[3] < min3) min3 = s[3];
        if (s[4] < min4) min4 = s[4];
        if (s[0] > max0) max0 = s[0];
        if (s[1] > max1) max1 = s[1];
        if (s[2] > max2) max2 = s[2];
        if (s[3] > max3) max3 = s[3];
        if (s[4] > max4) max4 = s[4];
    }

    min_values[0] = min0; min_values[1] = min1; min_values[2] = min2;
    min_values[3] = min3; min_values[4] = min4;
    max_values[0] = max0; max_values[1] = max1; max_values[2] = max2;
    max_values[3] = max3; max_values[4] = max4;
}

void build_random_forest_synthetic_omp(
    RandomForest* forest,
    const float min_values[NUM_FEATURES],
    const float max_values[NUM_FEATURES])
{
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < NUM_TREES; t++) {
        build_tree_stateless(forest, t, FOREST_SEED_DEFAULT, min_values, max_values);
    }
}

void predict_random_forest_omp(
    const float* features,
    const RandomForest* forest,
    float* predictions,
    int num_samples)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_samples; i++) {
        predictions[i] = predict_sample_cpu(&features[(size_t)i * NUM_FEATURES], forest);
    }
}

int main(int argc, char* argv[])
{
    CliOptions opt;
    if (cli_parse(argc, argv, &opt)) {
        return EXIT_SUCCESS;
    }

    printf("=== RANDOM FOREST - INFERENCIA OPENMP ===\n\n");
    printf("Arquivo: %s\n", opt.csv_filename);

    int base_samples = 0;
    float* base_features = read_csv_data(opt.csv_filename, &base_samples);
    if (!base_features) {
        return EXIT_FAILURE;
    }

    int num_samples = base_samples;
    float* features = base_features;
    if (opt.expand_n > 0) {
        features = expand_features_cyclic(base_features, base_samples, opt.expand_n);
        free(base_features);
        if (!features) {
            fprintf(stderr, "Erro de alocação ao expandir amostras.\n");
            return EXIT_FAILURE;
        }
        num_samples = opt.expand_n;
    }

    printf("Amostras carregadas: %d\n", num_samples);

    float min_values[NUM_FEATURES];
    float max_values[NUM_FEATURES];
    double t_ranges_ini = forest_wall_time_sec();
    compute_feature_ranges_omp(features, num_samples, min_values, max_values);
    double t_ranges_fim = forest_wall_time_sec();
    print_feature_ranges(min_values, max_values);

    float* predictions = (float*)malloc((size_t)num_samples * sizeof(float));
    RandomForest* forest = (RandomForest*)malloc(sizeof(RandomForest));
    double* times = (double*)malloc((size_t)opt.repeat * sizeof(double));
    if (!predictions || !forest || !times) {
        fprintf(stderr, "Erro de alocação de memória.\n");
        free(features); free(predictions); free(forest); free(times);
        return EXIT_FAILURE;
    }

    double t_build_ini = forest_wall_time_sec();
    if (opt.model_filename) {
        if (!load_forest_bin(opt.model_filename, forest)) {
            free(features); free(predictions); free(forest); free(times);
            return EXIT_FAILURE;
        }
        printf("\nModelo carregado: %s\n", opt.model_filename);
    } else if (opt.legacy_rand) {
        build_random_forest_legacy(forest, min_values, max_values);
        printf("\nModelo: floresta sintetica (rand legado, seed=%u)\n", FOREST_SEED_DEFAULT);
    } else {
        build_random_forest_synthetic_omp(forest, min_values, max_values);
        printf("\nModelo: floresta sintetica paralela deterministica (seed=%u)\n", FOREST_SEED_DEFAULT);
    }
    double t_build_fim = forest_wall_time_sec();

    int num_threads = omp_get_max_threads();

    for (int w = 0; w < opt.warmup; w++) {
        predict_random_forest_omp(features, forest, predictions, num_samples);
    }

    for (int r = 0; r < opt.repeat; r++) {
        double start = forest_wall_time_sec();
        predict_random_forest_omp(features, forest, predictions, num_samples);
        double end = forest_wall_time_sec();
        times[r] = end - start;
    }

    double elapsed = median_double(times, opt.repeat);
    printf("\nThreads OpenMP: %d\n", num_threads);
    printf("Tempo faixas (OpenMP): %.6f segundos\n", t_ranges_fim - t_ranges_ini);
    printf("Tempo construcao/carga (OpenMP): %.6f segundos\n", t_build_fim - t_build_ini);
    printf("Tempo de inferencia: %.6f segundos\n", elapsed);

    if (opt.dump_filename) {
        dump_predictions_bin(opt.dump_filename, predictions, num_samples);
    }

    print_rain_forecast(predictions, num_samples, opt.forecast_tail);

    free(features);
    free(predictions);
    free(forest);
    free(times);
    return EXIT_SUCCESS;
}
