/*
 * MeteoPel - Versão Nativa em CUDA C++
 *
 * Implementa dois kernels selecionáveis por linha de comando (--kernel):
 *   1. global (padrão): 1 thread por amostra lendo RandomForest na memória global
 *      com __restrict__ (cache read-only). Mantém fidelidade direta ao relatório base.
 *   2. shared (otimizado): carrega cooperativamente toda a RandomForest (30.400 bytes)
 *      para a memória __shared__ do bloco (< 48 KB) e lê as entradas em layout SoA
 *      (Structure of Arrays) para acesso coalescido pelos warps, usando Pinned Memory.
 *
 * Compilação: nvcc -O3 -arch=native -o meteo_cuda src/meteo_cuda.cu
 */

#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <time.h>
#include "../include/forest_common.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "Erro CUDA em %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    } while (0)

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

/* Kernel 1: Memória global (AoS) - baseline direto da versão OpenMP */
__global__ void inferir_global(
    const float* __restrict__ features,
    const RandomForest* __restrict__ forest,
    float* __restrict__ predictions,
    int num_samples)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_samples) return;

    const float* sample = features + (size_t)i * NUM_FEATURES;
    float total = 0.0f;

    for (int t = 0; t < NUM_TREES; ++t) {
        int current_node = forest->tree_roots[t];
        while (true) {
            Node node = forest->nodes[current_node];
            if (node.feature_index == -1) {
                total += node.value;
                break;
            }
            float val = sample[node.feature_index];
            current_node = (val <= node.threshold) ? node.left_child : node.right_child;
        }
    }

    predictions[i] = total / (float)NUM_TREES;
}

/* Kernel 2: Memória compartilhada (__shared__ para toda a floresta de 30,4 KB) + entradas SoA */
__global__ void inferir_shared_soa(
    const float* __restrict__ features_soa,
    const RandomForest* __restrict__ forest_global,
    float* __restrict__ predictions,
    int num_samples)
{
    __shared__ RandomForest s_forest;

    /* Carga cooperativa da RandomForest (30.400 bytes = 7.600 inteiros de 32 bits) */
    const uint32_t* src = reinterpret_cast<const uint32_t*>(forest_global);
    uint32_t* dst = reinterpret_cast<uint32_t*>(&s_forest);
    const int total_words = (int)(sizeof(RandomForest) / sizeof(uint32_t));
    for (int idx = threadIdx.x; idx < total_words; idx += blockDim.x) {
        dst[idx] = src[idx];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_samples) return;

    /* Leituras coalescidas no warp (SoA) mantidas em registradores */
    float f[NUM_FEATURES];
    f[0] = features_soa[(size_t)0 * num_samples + i];
    f[1] = features_soa[(size_t)1 * num_samples + i];
    f[2] = features_soa[(size_t)2 * num_samples + i];
    f[3] = features_soa[(size_t)3 * num_samples + i];
    f[4] = features_soa[(size_t)4 * num_samples + i];

    float total = 0.0f;
    for (int t = 0; t < NUM_TREES; ++t) {
        int current_node = s_forest.tree_roots[t];
        while (true) {
            Node node = s_forest.nodes[current_node];
            if (node.feature_index == -1) {
                total += node.value;
                break;
            }
            float val = f[node.feature_index];
            current_node = (val <= node.threshold) ? node.left_child : node.right_child;
        }
    }

    predictions[i] = total / (float)NUM_TREES;
}

static void transpose_aos_to_soa(const float* aos, float* soa, int num_samples)
{
    for (int i = 0; i < num_samples; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            soa[(size_t)f * num_samples + i] = aos[(size_t)i * NUM_FEATURES + f];
        }
    }
}

int main(int argc, char* argv[])
{
    CliOptions opt;
    if (cli_parse(argc, argv, &opt)) {
        return EXIT_SUCCESS;
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "Erro: nenhuma GPU CUDA encontrada.\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int use_shared = (strcmp(opt.kernel_mode, "shared") == 0);
    printf("=== RANDOM FOREST - INFERENCIA CUDA (%s) ===\n\n",
           use_shared ? "SHARED + SoA + PINNED" : "GLOBAL");
    printf("GPU: %s (CC %d.%d | SMs: %d | Shared/Bloco: %zu KB)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.sharedMemPerBlock / 1024);
    printf("Arquivo: %s\n", opt.csv_filename);

    int base_samples = 0;
    float* base_features = read_csv_data(opt.csv_filename, &base_samples);
    if (!base_features) {
        return EXIT_FAILURE;
    }

    int num_samples = base_samples;
    float* features_aos = base_features;
    if (opt.expand_n > 0) {
        features_aos = expand_features_cyclic(base_features, base_samples, opt.expand_n);
        free(base_features);
        if (!features_aos) {
            fprintf(stderr, "Erro de alocação ao expandir amostras.\n");
            return EXIT_FAILURE;
        }
        num_samples = opt.expand_n;
    }

    printf("Amostras carregadas: %d\n", num_samples);

    float min_values[NUM_FEATURES];
    float max_values[NUM_FEATURES];
    compute_feature_ranges_seq(features_aos, num_samples, min_values, max_values);
    print_feature_ranges(min_values, max_values);

    /* Aloca buffers de host em Pinned Memory (cudaHostAlloc) para máxima banda PCIe */
    size_t feat_bytes = (size_t)num_samples * NUM_FEATURES * sizeof(float);
    size_t pred_bytes = (size_t)num_samples * sizeof(float);
    float* h_features = NULL;
    float* h_predictions = NULL;
    RandomForest* h_forest = NULL;

    CUDA_CHECK(cudaHostAlloc(&h_features, feat_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_predictions, pred_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_forest, sizeof(RandomForest), cudaHostAllocDefault));

    if (use_shared) {
        transpose_aos_to_soa(features_aos, h_features, num_samples);
    } else {
        memcpy(h_features, features_aos, feat_bytes);
    }
    free(features_aos);

    if (opt.model_filename) {
        if (!load_forest_bin(opt.model_filename, h_forest)) {
            cudaFreeHost(h_features); cudaFreeHost(h_predictions); cudaFreeHost(h_forest);
            return EXIT_FAILURE;
        }
        printf("\nModelo carregado: %s\n", opt.model_filename);
    } else if (opt.legacy_rand) {
        build_random_forest_legacy(h_forest, min_values, max_values);
        printf("\nModelo: floresta sintetica (rand legado, seed=%u)\n", FOREST_SEED_DEFAULT);
    } else {
        build_random_forest_stateless_seq(h_forest, min_values, max_values);
        printf("\nModelo: floresta sintetica deterministica (seed=%u)\n", FOREST_SEED_DEFAULT);
    }

    float* d_features = NULL;
    float* d_predictions = NULL;
    RandomForest* d_forest = NULL;
    CUDA_CHECK(cudaMalloc(&d_features, feat_bytes));
    CUDA_CHECK(cudaMalloc(&d_predictions, pred_bytes));
    CUDA_CHECK(cudaMalloc(&d_forest, sizeof(RandomForest)));

    int block_size = opt.block_size;
    int grid_size = (num_samples + block_size - 1) / block_size;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    double* kernel_times = (double*)malloc((size_t)opt.repeat * sizeof(double));
    double* total_times = (double*)malloc((size_t)opt.repeat * sizeof(double));

    int total_runs = opt.warmup + opt.repeat;
    for (int run = 0; run < total_runs; run++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = wall_time_sec();

        CUDA_CHECK(cudaMemcpy(d_features, h_features, feat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_forest, h_forest, sizeof(RandomForest), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(ev_start));
        if (use_shared) {
            inferir_shared_soa<<<grid_size, block_size>>>(d_features, d_forest, d_predictions, num_samples);
        } else {
            inferir_global<<<grid_size, block_size>>>(d_features, d_forest, d_predictions, num_samples);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_stop));

        CUDA_CHECK(cudaMemcpy(h_predictions, d_predictions, pred_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        double t1 = wall_time_sec();

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));

        if (run >= opt.warmup) {
            int idx = run - opt.warmup;
            kernel_times[idx] = (double)kernel_ms * 1e-3;
            total_times[idx] = t1 - t0;
        }
    }

    double med_kernel = median_double(kernel_times, opt.repeat);
    double med_total = median_double(total_times, opt.repeat);

    printf("\nGrade CUDA: %d blocos x %d threads/bloco (kernel: %s)\n",
           grid_size, block_size, use_shared ? "shared+SoA" : "global");
    printf("Tempo de inferencia (kernel CUDA): %.6f segundos\n", med_kernel);
    printf("Tempo de inferencia (total H2D+kernel+D2H): %.6f segundos\n", med_total);

    if (opt.dump_filename) {
        dump_predictions_bin(opt.dump_filename, h_predictions, num_samples);
    }

    print_rain_forecast(h_predictions, num_samples, opt.forecast_tail);

    free(kernel_times);
    free(total_times);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_features);
    cudaFree(d_predictions);
    cudaFree(d_forest);
    cudaFreeHost(h_features);
    cudaFreeHost(h_predictions);
    cudaFreeHost(h_forest);

    return EXIT_SUCCESS;
}
