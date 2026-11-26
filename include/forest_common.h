#ifndef FOREST_COMMON_H
#define FOREST_COMMON_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <time.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

static inline double forest_wall_time_sec(void)
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

#define NUM_FEATURES 5
#define NUM_TREES 100
#define MAX_NODES_PER_TREE 15
#define NUM_INTERNAL_NODES 7
#define FIRST_LEAF 7
#define MAX_LINE_LENGTH 256
#define FOREST_SEED_DEFAULT 42u

/*
 * Cada árvore é uma árvore binária completa com 15 nós:
 *
 *              0
 *          /       \
 *         1         2
 *       /  \       /  \
 *      3    4     5    6
 *     / \  / \   / \  / \
 *    7  8 9  10 11 12 13 14
 *
 * Nós 0..6  : nós internos
 * Nós 7..14 : folhas
 */

typedef struct {
    int32_t feature_index;   /* -1 indica folha */
    float threshold;
    float value;
    int32_t left_child;
    int32_t right_child;
} Node;

typedef struct {
    Node nodes[NUM_TREES * MAX_NODES_PER_TREE];
    int32_t tree_roots[NUM_TREES];
} RandomForest;

#if defined(__cplusplus)
static_assert(sizeof(Node) == 20, "Node deve ocupar exatamente 20 bytes");
static_assert(sizeof(RandomForest) == 30400, "RandomForest deve ocupar exatamente 30400 bytes");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(Node) == 20, "Node deve ocupar exatamente 20 bytes");
_Static_assert(sizeof(RandomForest) == 30400, "RandomForest deve ocupar exatamente 30400 bytes");
#endif

typedef struct {
    const char* csv_filename;
    const char* model_filename;
    const char* dump_filename;
    const char* kernel_mode; /* apenas CUDA: "global" ou "shared" */
    int repeat;
    int warmup;
    int forecast_tail;       /* 1 = exibe últimas horas (previsão futura) */
    int legacy_rand;         /* 1 = usa srand(42)+rand() clássico sequencial */
    int block_size;          /* apenas CUDA: threads por bloco */
    int expand_n;            /* 0 = usa N do CSV; >0 = replica ciclicamente até expand_n */
} CliOptions;

static inline void cli_init_defaults(CliOptions* opt)
{
    opt->csv_filename = "dados_pelotas.csv";
    opt->model_filename = NULL;
    opt->dump_filename = NULL;
    opt->kernel_mode = "global";
    opt->repeat = 1;
    opt->warmup = 0;
    opt->forecast_tail = 0;
    opt->legacy_rand = 0;
    opt->block_size = 256;
    opt->expand_n = 0;
}

static inline int cli_parse(int argc, char* argv[], CliOptions* opt)
{
    cli_init_defaults(opt);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            opt->model_filename = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            opt->dump_filename = argv[++i];
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            opt->kernel_mode = argv[++i];
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            opt->repeat = atoi(argv[++i]);
            if (opt->repeat < 1) opt->repeat = 1;
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            opt->warmup = atoi(argv[++i]);
            if (opt->warmup < 0) opt->warmup = 0;
        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            opt->block_size = atoi(argv[++i]);
            if (opt->block_size < 32) opt->block_size = 32;
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            opt->expand_n = atoi(argv[++i]);
            if (opt->expand_n < 0) opt->expand_n = 0;
        } else if (strcmp(argv[i], "--forecast") == 0) {
            opt->forecast_tail = 1;
        } else if (strcmp(argv[i], "--legacy-rand") == 0) {
            opt->legacy_rand = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Uso: %s [arquivo.csv] [opcoes]\n", argv[0]);
            printf("  --model <arq.rf>        Carrega floresta binaria treinada (30400 bytes)\n");
            printf("  --kernel global|shared  Seleciona kernel CUDA (padrao: global)\n");
            printf("  --block-size <B>        Threads por bloco CUDA (padrao: 256)\n");
            printf("  --n <N>                 Replica amostras ciclicamente ate N (benchmark)\n");
            printf("  --warmup <W>            Rodadas de aquecimento antes de medir (padrao: 0)\n");
            printf("  --repeat <R>            Repeticoes cronometradas para mediana (padrao: 1)\n");
            printf("  --forecast              Exibe as ultimas 24h (janela futura) em vez das 5 iniciais\n");
            printf("  --legacy-rand           Usa srand(42)+rand() sequencial classico\n");
            printf("  --dump <preds.bin>      Grava vetor float32 de previsoes para validacao\n");
            return 1;
        } else if (argv[i][0] != '-') {
            opt->csv_filename = argv[i];
        } else {
            fprintf(stderr, "Aviso: opcao desconhecida ignorada: %s\n", argv[i]);
        }
    }
    return 0;
}

/* ================================================================
 * Leitura e expansão opcional do CSV meteorológico
 * ================================================================ */

static inline float* read_csv_data(const char* filename, int* out_num_samples)
{
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Erro ao abrir o arquivo CSV");
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    int line_count = 0;

    if (fgets(line, sizeof(line), file) == NULL) {
        fprintf(stderr, "Erro: arquivo CSV vazio.\n");
        fclose(file);
        return NULL;
    }

    while (fgets(line, sizeof(line), file)) {
        float f0, f1, f2, f3, f4;
        if (sscanf(line, "%f,%f,%f,%f,%f", &f0, &f1, &f2, &f3, &f4) == NUM_FEATURES) {
            line_count++;
        }
    }

    if (line_count == 0) {
        fprintf(stderr, "Erro: nenhuma amostra válida encontrada no CSV.\n");
        fclose(file);
        return NULL;
    }

    float* features = (float*)malloc((size_t)line_count * NUM_FEATURES * sizeof(float));
    if (!features) {
        fprintf(stderr, "Erro de alocação de memória.\n");
        fclose(file);
        return NULL;
    }

    rewind(file);
    if (fgets(line, sizeof(line), file) == NULL) {
        free(features);
        fclose(file);
        return NULL;
    }

    int sample = 0;
    while (fgets(line, sizeof(line), file) && sample < line_count) {
        float f0, f1, f2, f3, f4;
        if (sscanf(line, "%f,%f,%f,%f,%f", &f0, &f1, &f2, &f3, &f4) == NUM_FEATURES) {
            features[(size_t)sample * NUM_FEATURES + 0] = f0;
            features[(size_t)sample * NUM_FEATURES + 1] = f1;
            features[(size_t)sample * NUM_FEATURES + 2] = f2;
            features[(size_t)sample * NUM_FEATURES + 3] = f3;
            features[(size_t)sample * NUM_FEATURES + 4] = f4;
            sample++;
        }
    }

    fclose(file);
    *out_num_samples = sample;
    return features;
}

static inline float* expand_features_cyclic(const float* base, int base_n, int target_n)
{
    float* out = (float*)malloc((size_t)target_n * NUM_FEATURES * sizeof(float));
    if (!out) return NULL;
    for (int i = 0; i < target_n; i++) {
        int src = i % base_n;
        memcpy(&out[(size_t)i * NUM_FEATURES],
               &base[(size_t)src * NUM_FEATURES],
               NUM_FEATURES * sizeof(float));
    }
    return out;
}

/* ================================================================
 * Persistência binária da floresta (ABI de 30.400 bytes)
 * ================================================================ */

static inline int load_forest_bin(const char* filename, RandomForest* forest)
{
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Erro ao abrir arquivo de modelo (.rf)");
        return 0;
    }
    size_t read_items = fread(forest, sizeof(RandomForest), 1, f);
    fclose(f);
    if (read_items != 1) {
        fprintf(stderr, "Erro: arquivo de modelo '%s' incompleto (esperados %zu bytes).\n",
                filename, sizeof(RandomForest));
        return 0;
    }
    return 1;
}

static inline int dump_predictions_bin(const char* filename, const float* predictions, int n)
{
    FILE* f = fopen(filename, "wb");
    if (!f) {
        perror("Erro ao gravar arquivo de previsoes");
        return 0;
    }
    size_t written = fwrite(predictions, sizeof(float), (size_t)n, f);
    fclose(f);
    return written == (size_t)n;
}

/* ================================================================
 * PRNG determinístico stateless (thread-safe para OpenMP/MPI/CUDA)
 * ================================================================ */

static inline uint32_t hash_u32(uint32_t x)
{
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    return x ^ (x >> 16);
}

static inline uint32_t stateless_u32(uint32_t seed, int tree_id, int slot)
{
    uint32_t key = seed ^ ((uint32_t)tree_id * 0x9e3779b1u) ^ ((uint32_t)slot * 0x85ebca77u);
    return hash_u32(key);
}

static inline float stateless_float(uint32_t seed, int tree_id, int slot,
                                    float min_value, float max_value)
{
    uint32_t bits = stateless_u32(seed, tree_id, slot) >> 8; /* 24 bits de mantissa */
    float r = (float)bits / 16777215.0f;
    return min_value + r * (max_value - min_value);
}

/* ================================================================
 * Faixas observadas e construção da floresta (versão sequencial)
 * ================================================================ */

static inline void compute_feature_ranges_seq(
    const float* features,
    int num_samples,
    float min_values[NUM_FEATURES],
    float max_values[NUM_FEATURES])
{
    for (int f = 0; f < NUM_FEATURES; f++) {
        min_values[f] = FLT_MAX;
        max_values[f] = -FLT_MAX;
    }

    for (int i = 0; i < num_samples; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            float value = features[(size_t)i * NUM_FEATURES + f];
            if (value < min_values[f]) min_values[f] = value;
            if (value > max_values[f]) max_values[f] = value;
        }
    }
}

static inline float legacy_random_float(float min_value, float max_value)
{
    float r = (float)rand() / (float)RAND_MAX;
    return min_value + r * (max_value - min_value);
}

static inline void build_random_forest_legacy(
    RandomForest* forest,
    const float min_values[NUM_FEATURES],
    const float max_values[NUM_FEATURES])
{
    srand(FOREST_SEED_DEFAULT);
    int node_counter = 0;

    for (int t = 0; t < NUM_TREES; t++) {
        int tree_start = node_counter;
        forest->tree_roots[t] = tree_start;

        for (int n = 0; n < NUM_INTERNAL_NODES; n++) {
            int feature = rand() % NUM_FEATURES;
            float threshold = legacy_random_float(min_values[feature], max_values[feature]);
            int left = tree_start + (2 * n + 1);
            int right = tree_start + (2 * n + 2);
            Node nd = { feature, threshold, 0.0f, left, right };
            forest->nodes[tree_start + n] = nd;
        }

        for (int n = FIRST_LEAF; n < MAX_NODES_PER_TREE; n++) {
            float prediction = legacy_random_float(0.0f, 35.0f);
            Node nd = { -1, 0.0f, prediction, -1, -1 };
            forest->nodes[tree_start + n] = nd;
        }

        node_counter += MAX_NODES_PER_TREE;
    }
}

static inline void build_tree_stateless(
    RandomForest* forest,
    int t,
    uint32_t seed,
    const float min_values[NUM_FEATURES],
    const float max_values[NUM_FEATURES])
{
    int tree_start = t * MAX_NODES_PER_TREE;
    forest->tree_roots[t] = tree_start;

    for (int n = 0; n < NUM_INTERNAL_NODES; n++) {
        int feature = (int)(stateless_u32(seed, t, n * 2) % NUM_FEATURES);
        float threshold = stateless_float(seed, t, n * 2 + 1,
                                          min_values[feature], max_values[feature]);
        int left = tree_start + (2 * n + 1);
        int right = tree_start + (2 * n + 2);
        Node nd = { feature, threshold, 0.0f, left, right };
        forest->nodes[tree_start + n] = nd;
    }

    for (int n = FIRST_LEAF; n < MAX_NODES_PER_TREE; n++) {
        float prediction = stateless_float(seed, t, 100 + n, 0.0f, 35.0f);
        Node nd = { -1, 0.0f, prediction, -1, -1 };
        forest->nodes[tree_start + n] = nd;
    }
}

static inline void build_random_forest_stateless_seq(
    RandomForest* forest,
    const float min_values[NUM_FEATURES],
    const float max_values[NUM_FEATURES])
{
    for (int t = 0; t < NUM_TREES; t++) {
        build_tree_stateless(forest, t, FOREST_SEED_DEFAULT, min_values, max_values);
    }
}

/* ================================================================
 * Inferência de uma única amostra e interpretação textual
 * ================================================================ */

static inline float predict_sample_cpu(const float* sample, const RandomForest* forest)
{
    float total_prediction = 0.0f;

    for (int t = 0; t < NUM_TREES; t++) {
        int current_node = forest->tree_roots[t];
        while (1) {
            Node node = forest->nodes[current_node];
            if (node.feature_index == -1) {
                total_prediction += node.value;
                break;
            }
            float feature_value = sample[node.feature_index];
            current_node = (feature_value <= node.threshold)
                               ? node.left_child
                               : node.right_child;
        }
    }

    return total_prediction / (float)NUM_TREES;
}

static inline const char* classificar_risco_chuva(float mm_previstos)
{
    if (mm_previstos < 1.0f)  return "Tempo Seco / Estável";
    if (mm_previstos < 5.0f)  return "Chuva Leve / Garoa";
    if (mm_previstos < 15.0f) return "Chuva Moderada";
    if (mm_previstos < 30.0f) return "ALERTA: Chuva Forte";
    return "PERIGO: Tempestade Severa";
}

static inline int cmp_double_asc(const void* a, const void* b)
{
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static inline double median_double(double* vals, int n)
{
    qsort(vals, (size_t)n, sizeof(double), cmp_double_asc);
    if (n % 2 == 1) return vals[n / 2];
    return 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
}

static inline void print_feature_ranges(
    const float min_values[NUM_FEATURES],
    const float max_values[NUM_FEATURES])
{
    printf("\nFaixas observadas:\n");
    for (int f = 0; f < NUM_FEATURES; f++) {
        printf("Feature %d: %.2f a %.2f\n", f, min_values[f], max_values[f]);
    }
}

static inline void print_rain_forecast(
    const float* predictions,
    int num_samples,
    int forecast_tail)
{
    if (forecast_tail) {
        int count = num_samples < 24 ? num_samples : 24;
        int start = num_samples - count;
        printf("\n--- PREVISÃO DE CHUVA PARA AS PRÓXIMAS %d HORAS (PELOTAS) ---\n", count);
        for (int i = 0; i < count; i++) {
            float mm = predictions[start + i];
            printf("Previsão [+%02dh] (amostra %d) -> Estimativa: %5.2f mm/h | Status: %s\n",
                   i + 1, start + i, mm, classificar_risco_chuva(mm));
        }
    } else {
        int print_limit = num_samples < 5 ? num_samples : 5;
        printf("\n--- PREVISÃO DE CHUVA PARA A PRÓXIMA HORA (PELOTAS) ---\n");
        for (int i = 0; i < print_limit; i++) {
            float mm = predictions[i];
            printf("Hora [%d] -> Estimativa: %5.2f mm/h | Status: %s\n",
                   i, mm, classificar_risco_chuva(mm));
        }
    }
}

#endif /* FOREST_COMMON_H */
