/* Stand-in libcuda.so.1 for e2e: SM count and shared memory for the same UUIDs as nvml_shim.
   FAKE_CUDA_SMS (default 170) and FAKE_CUDA_SMEM (default 101376) set the attributes. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MULTIPROCESSOR_COUNT 16
#define MAX_SHARED_MEMORY_PER_BLOCK_OPTIN 97
#define MAX_GPUS 16

static int n_devices;

static int device_count(void) {
    const char *spec = getenv("FAKE_NVML");
    char *copy, *entry, *save = NULL;
    int n = 0;
    if (!spec || !*spec) return 0;
    copy = strdup(spec);
    for (entry = strtok_r(copy, ";", &save); entry && n < MAX_GPUS; entry = strtok_r(NULL, ";", &save)) {
        if (strchr(entry, '|')) n++;
    }
    free(copy);
    return n;
}

static int sms_of(void) {
    const char *v = getenv("FAKE_CUDA_SMS");
    return v && *v ? atoi(v) : 170;
}

static int smem_of(void) {
    const char *v = getenv("FAKE_CUDA_SMEM");
    return v && *v ? atoi(v) : 101376;
}

int cuInit(unsigned flags) {
    (void)flags;
    n_devices = device_count();
    return 0;
}

int cuDeviceGetCount(int *count) {
    *count = n_devices;
    return 0;
}

int cuDeviceGet(int *device, int ordinal) {
    if (ordinal < 0 || ordinal >= n_devices) return 1;
    *device = ordinal;
    return 0;
}

int cuDeviceGetUuid(unsigned char *uuid, int device) {
    /* Match nvml_shim: GPU-0000000{i}-0000-0000-0000-000000000000 */
    char hex[33];
    int i;
    if (device < 0 || device >= n_devices) return 1;
    snprintf(hex, sizeof hex, "%08x000000000000000000000000", (unsigned)device);
    for (i = 0; i < 16; i++) {
        unsigned v = 0;
        sscanf(hex + 2 * i, "%2x", &v);
        uuid[i] = (unsigned char)v;
    }
    return 0;
}

int cuDeviceGetAttribute(int *pi, int attrib, int device) {
    if (device < 0 || device >= n_devices) return 1;
    if (attrib == MULTIPROCESSOR_COUNT) {
        *pi = sms_of();
        return 0;
    }
    if (attrib == MAX_SHARED_MEMORY_PER_BLOCK_OPTIN) {
        *pi = smem_of();
        return 0;
    }
    *pi = 0;
    return 0;
}
