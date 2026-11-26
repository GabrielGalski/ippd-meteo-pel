/*
 * MeteoPel - Versão Sequencial de Referência (CPU 1 thread)
 * Compilação: gcc -O3 -o meteo src/meteo.c -lm
 */

#define _POSIX_C_SOURCE 200809L
#ifdef _WIN32
#include <windows.h>
#endif
#include <time.h>
#include "../include/forest_common.h"

static double wall_time_sec(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#elif defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

static void predict_random_forest_seq(
    const float* features,
    const RandomForest* forest,
    float* predictions,
    int num_samples)
{
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

    printf("=== RANDOM FOREST - INFERENCIA SEQUENCIAL ===\n\n");
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
    double t_ranges_ini = wall_time_sec();
    compute_feature_ranges_seq(features, num_samples, min_values, max_values);
    double t_ranges_fim = wall_time_sec();
    print_feature_ranges(min_values, max_values);

    float* predictions = (float*)malloc((size_t)num_samples * sizeof(float));
    RandomForest* forest = (RandomForest*)malloc(sizeof(RandomForest));
    double* times = (double*)malloc((size_t)opt.repeat * sizeof(double));
    if (!predictions || !forest || !times) {
        fprintf(stderr, "Erro de alocação de memória.\n");
        free(features);
        free(predictions);
        free(forest);
        free(times);
        return EXIT_FAILURE;
    }

    double t_build_ini = wall_time_sec();
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
        build_random_forest_stateless_seq(forest, min_values, max_values);
        printf("\nModelo: floresta sintetica deterministica (seed=%u)\n", FOREST_SEED_DEFAULT);
    }
    double t_build_fim = wall_time_sec();

    for (int w = 0; w < opt.warmup; w++) {
        predict_random_forest_seq(features, forest, predictions, num_samples);
    }

    for (int r = 0; r < opt.repeat; r++) {
        double start = wall_time_sec();
        predict_random_forest_seq(features, forest, predictions, num_samples);
        double end = wall_time_sec();
        times[r] = end - start;
    }

    double elapsed = median_double(times, opt.repeat);
    printf("\nModo: Sequencial (1 thread)\n");
    printf("Tempo faixas: %.6f segundos\n", t_ranges_fim - t_ranges_ini);
    printf("Tempo construcao/carga: %.6f segundos\n", t_build_fim - t_build_ini);
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
