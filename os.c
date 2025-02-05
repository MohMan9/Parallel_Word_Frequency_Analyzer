#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <fcntl.h>

#define MAX_WORD_LENGTH 100
#define MAX_UNIQUE_WORDS 10000
#define MAX_TOTAL_WORDS 17005208
#define NUM_TOP_WORDS 10

typedef struct {
    char word[MAX_WORD_LENGTH];
    int count;
} WordFreq;

typedef struct {
    WordFreq words[MAX_UNIQUE_WORDS];
    int count;
} SharedMemory;

typedef struct {
    char** words;
    int start;
    int end;
    WordFreq* local_freq;
    int* local_count;
} ThreadArgs;

double time_diff_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

int compare(const void *a, const void *b) {
    return ((WordFreq *)b)->count - ((WordFreq *)a)->count;
}

void clean_word(char *word) {
    char *src = word, *dst = word;
    while (*src) {
        if (isalnum(*src)) {
            *dst = tolower(*src);
            dst++;
        }
        src++;
    }
    *dst = '\0';
}

void print_results(const char* approach, WordFreq* words, int count) {
    printf("\n%s Results:\n", approach);
    printf("Unique words found: %d\n", count);
    printf("Top %d most frequent words:\n", NUM_TOP_WORDS);
    for (int i = 0; i < NUM_TOP_WORDS && i < count; i++) {
        printf("%d. %s: %d\n", i + 1, words[i].word, words[i].count);
    }
    printf("----------------------------------------\n");
}

void naive_count(char** words, int total_words, double* serial_time) {
    struct timespec start_serial, end_serial, start_parallel, end_parallel;

    // Start timing serial portion (memory allocation)
    clock_gettime(CLOCK_MONOTONIC, &start_serial);
    WordFreq* freq = calloc(MAX_UNIQUE_WORDS, sizeof(WordFreq));
    int unique_count = 0;
    clock_gettime(CLOCK_MONOTONIC, &end_serial);
    double alloc_time = time_diff_ms(start_serial, end_serial);

    // Parallel portion (word counting)
    clock_gettime(CLOCK_MONOTONIC, &start_parallel);
    for (int i = 0; i < total_words; i++) {
        int found = 0;
        for (int j = 0; j < unique_count; j++) {
            if (strcmp(freq[j].word, words[i]) == 0) {
                freq[j].count++;
                found = 1;
                break;
            }
        }
        if (!found && unique_count < MAX_UNIQUE_WORDS) {
            strncpy(freq[unique_count].word, words[i], MAX_WORD_LENGTH - 1);
            freq[unique_count].word[MAX_WORD_LENGTH - 1] = '\0';
            freq[unique_count].count = 1;
            unique_count++;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end_parallel);
    double parallel_time = time_diff_ms(start_parallel, end_parallel);

    // Start timing serial portion (sorting)
    clock_gettime(CLOCK_MONOTONIC, &start_serial);
    qsort(freq, unique_count, sizeof(WordFreq), compare);
    clock_gettime(CLOCK_MONOTONIC, &end_serial);
    double sort_time = time_diff_ms(start_serial, end_serial);

    *serial_time = alloc_time + sort_time;

    print_results("Naive Approach", freq, unique_count);
    printf("Allocation time: %.2f ms\n", alloc_time);
    printf("Parallel processing time: %.2f ms\n", parallel_time);
    printf("Sorting time: %.2f ms\n", sort_time);

    free(freq);
}

void multiprocess_count(char** words, int total_words, int num_processes, double* serial_time) {
    struct timespec start_serial, end_serial, start_parallel, end_parallel;

    // Start timing serial portion (shared memory setup)
    clock_gettime(CLOCK_MONOTONIC, &start_serial);
    SharedMemory* shared = mmap(NULL, sizeof(SharedMemory),
                              PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    if (shared == MAP_FAILED) {
        perror("mmap failed");
        return;
    }

    memset(shared, 0, sizeof(SharedMemory));
    clock_gettime(CLOCK_MONOTONIC, &end_serial);
    double setup_time = time_diff_ms(start_serial, end_serial);

    int words_per_process = total_words / num_processes;
    int remainder = total_words % num_processes;

    // Parallel portion (word counting)
    clock_gettime(CLOCK_MONOTONIC, &start_parallel);
    for (int i = 0; i < num_processes; i++) {
        pid_t pid = fork();

        if (pid < 0) {
            perror("Fork failed");
            exit(1);
        }

        if (pid == 0) {
            int start = i * words_per_process + (i < remainder ? i : remainder);
            int end = start + words_per_process + (i < remainder ? 1 : 0);

            WordFreq* local = calloc(MAX_UNIQUE_WORDS, sizeof(WordFreq));
            int local_count = 0;

            for (int j = start; j < end; j++) {
                int found = 0;
                for (int k = 0; k < local_count; k++) {
                    if (strcmp(local[k].word, words[j]) == 0) {
                        local[k].count++;
                        found = 1;
                        break;
                    }
                }
                if (!found && local_count < MAX_UNIQUE_WORDS) {
                    strncpy(local[local_count].word, words[j], MAX_WORD_LENGTH - 1);
                    local[local_count].word[MAX_WORD_LENGTH - 1] = '\0';
                    local[local_count].count = 1;
                    local_count++;
                }
            }

            for (int j = 0; j < local_count; j++) {
                int found = 0;
                for (int k = 0; k < shared->count; k++) {
                    if (strcmp(shared->words[k].word, local[j].word) == 0) {
                        shared->words[k].count += local[j].count;
                        found = 1;
                        break;
                    }
                }
                if (!found && shared->count < MAX_UNIQUE_WORDS) {
                    strncpy(shared->words[shared->count].word, local[j].word, MAX_WORD_LENGTH - 1);
                    shared->words[shared->count].word[MAX_WORD_LENGTH - 1] = '\0';
                    shared->words[shared->count].count = local[j].count;
                    shared->count++;
                }
            }

            free(local);
            exit(0);
        }
    }

    for (int i = 0; i < num_processes; i++) {
        wait(NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_parallel);
    double parallel_time = time_diff_ms(start_parallel, end_parallel);

    // Serial portion (sorting and cleanup)
    clock_gettime(CLOCK_MONOTONIC, &start_serial);
    qsort(shared->words, shared->count, sizeof(WordFreq), compare);
    clock_gettime(CLOCK_MONOTONIC, &end_serial);
    double sort_time = time_diff_ms(start_serial, end_serial);

    *serial_time = setup_time + sort_time;

    char approach[50];
    sprintf(approach, "Multiprocessing Approach (%d processes)", num_processes);
    print_results(approach, shared->words, shared->count);
    printf("Setup time: %.2f ms\n", setup_time);
    printf("Parallel processing time: %.2f ms\n", parallel_time);
    printf("Sorting time: %.2f ms\n", sort_time);

    munmap(shared, sizeof(SharedMemory));
}

void* thread_count(void* args) {
    ThreadArgs* thread_args = (ThreadArgs*)args;
    for (int i = thread_args->start; i < thread_args->end; i++) {
        int found = 0;
        for (int j = 0; j < *thread_args->local_count; j++) {
            if (strcmp(thread_args->local_freq[j].word, thread_args->words[i]) == 0) {
                thread_args->local_freq[j].count++;
                found = 1;
                break;
            }
        }
        if (!found && *thread_args->local_count < MAX_UNIQUE_WORDS) {
            strncpy(thread_args->local_freq[*thread_args->local_count].word,
                   thread_args->words[i], MAX_WORD_LENGTH - 1);
            thread_args->local_freq[*thread_args->local_count].word[MAX_WORD_LENGTH - 1] = '\0';
            thread_args->local_freq[*thread_args->local_count].count = 1;
            (*thread_args->local_count)++;
        }
    }
    return NULL;
}

void multithread_count(char** words, int total_words, int num_threads, double* serial_time) {
    struct timespec start_serial, end_serial, start_parallel, end_parallel;

    // Start timing serial portion (thread setup)
    clock_gettime(CLOCK_MONOTONIC, &start_serial);
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    ThreadArgs* args = malloc(num_threads * sizeof(ThreadArgs));
    WordFreq* freq = calloc(MAX_UNIQUE_WORDS, sizeof(WordFreq));
    int unique_count = 0;
    clock_gettime(CLOCK_MONOTONIC, &end_serial);
    double setup_time = time_diff_ms(start_serial, end_serial);

    int words_per_thread = total_words / num_threads;
    int remainder = total_words % num_threads;

    // Parallel portion (word counting)
    clock_gettime(CLOCK_MONOTONIC, &start_parallel);
    for (int i = 0; i < num_threads; i++) {
        int start = i * words_per_thread + (i < remainder ? i : remainder);
        int end = start + words_per_thread + (i < remainder ? 1 : 0);
        args[i] = (ThreadArgs){words, start, end, freq, &unique_count};
        pthread_create(&threads[i], NULL, thread_count, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_parallel);
    double parallel_time = time_diff_ms(start_parallel, end_parallel);

    // Serial portion (sorting and cleanup)
    clock_gettime(CLOCK_MONOTONIC, &start_serial);
    qsort(freq, unique_count, sizeof(WordFreq), compare);
    clock_gettime(CLOCK_MONOTONIC, &end_serial);
    double sort_time = time_diff_ms(start_serial, end_serial);

    *serial_time = setup_time + sort_time;

    char approach[50];
    sprintf(approach, "Multithreading Approach (%d threads)", num_threads);
    print_results(approach, freq, unique_count);
    printf("Setup time: %.2f ms\n", setup_time);
    printf("Parallel processing time: %.2f ms\n", parallel_time);
    printf("Sorting time: %.2f ms\n", sort_time);

    free(threads);
    free(args);
    free(freq);
}

int main() {
    FILE* file = fopen("mohammad.txt", "r");
    if (!file) {
        perror("Error opening file");
        return 1;
    }

    char** words = malloc(MAX_TOTAL_WORDS * sizeof(char*));
    char temp[MAX_WORD_LENGTH];
    int total_words = 0;

    while (fscanf(file, "%s", temp) == 1 && total_words < MAX_TOTAL_WORDS) {
        clean_word(temp);
        if (strlen(temp) > 0) {
            words[total_words] = malloc(MAX_WORD_LENGTH);
            strncpy(words[total_words], temp, MAX_WORD_LENGTH - 1);
            words[total_words][MAX_WORD_LENGTH - 1] = '\0';
            total_words++;
        }
    }
    fclose(file);

    printf("Total words read: %d\n", total_words);

    struct timespec start_total, end_total;
    int process_counts[] = {2, 4, 6, 8};
    double serial_time = 0;

    // Naive Approach
    clock_gettime(CLOCK_MONOTONIC, &start_total);
    naive_count(words, total_words, &serial_time);
    clock_gettime(CLOCK_MONOTONIC, &end_total);
    double total_time = time_diff_ms(start_total, end_total);
    printf("Naive Approach Total Time: %.2f ms\n", total_time);
    printf("Naive Approach Serial Time: %.2f ms (%.2f%%)\n",
           serial_time, (serial_time/total_time) * 100);
    printf("Naive Approach Parallel Time: %.2f ms (%.2f%%)\n",
           total_time - serial_time, ((total_time-serial_time)/total_time) * 100);

   // Multiprocessing
for (int i = 0; i < sizeof(process_counts) / sizeof(process_counts[0]); i++) {
    clock_gettime(CLOCK_MONOTONIC, &start_total);
    multiprocess_count(words, total_words, process_counts[i], &serial_time);
    clock_gettime(CLOCK_MONOTONIC, &end_total);

    total_time = time_diff_ms(start_total, end_total);

    printf("Multiprocessing (%d processes) Total Time: %.2f ms\n", process_counts[i], total_time);
    printf("Multiprocessing Serial Time: %.2f ms (%.2f%%)\n", serial_time, (serial_time / total_time) * 100);
    printf("Multiprocessing Parallel Time: %.2f ms (%.2f%%)\n", total_time - serial_time,
           ((total_time - serial_time) / total_time) * 100);
}

// Multithreading
for (int i = 0; i < sizeof(process_counts) / sizeof(process_counts[0]); i++) {
    clock_gettime(CLOCK_MONOTONIC, &start_total);
    multithread_count(words, total_words, process_counts[i], &serial_time);
    clock_gettime(CLOCK_MONOTONIC, &end_total);

    total_time = time_diff_ms(start_total, end_total);

    printf("Multithreading (%d threads) Total Time: %.2f ms\n", process_counts[i], total_time);
    printf("Multithreading Serial Time: %.2f ms (%.2f%%)\n", serial_time, (serial_time / total_time) * 100);
    printf("Multithreading Parallel Time: %.2f ms (%.2f%%)\n", total_time - serial_time,
           ((total_time - serial_time) / total_time) * 100);
}

// Clean up
for (int i = 0; i < total_words; i++) {
    free(words[i]);
}
free(words);

return 0;
}
