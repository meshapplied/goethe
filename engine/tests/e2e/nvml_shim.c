/* A stand-in libnvidia-ml.so.1: the GPUs come from FAKE_NVML="name|major.minor|total_mib|used_mib;..."; unset means no driver. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_GPUS 16

typedef struct { char name[96]; int major, minor; unsigned long long total, used; } Gpu;
typedef struct { unsigned long long total, free, used; } Memory;
typedef struct { char busIdLegacy[16]; unsigned domain, bus, device, pciDeviceId, pciSubSystemId; char busId[32]; } PciInfo;
typedef struct { unsigned gpu, memory; } Utilization;

static Gpu gpus[MAX_GPUS];
static unsigned count;

static int load(void) {
    const char *spec = getenv("FAKE_NVML");
    char *copy, *entry, *save = NULL;
    if (!spec || !*spec) return 9;
    copy = strdup(spec);
    count = 0;
    for (entry = strtok_r(copy, ";", &save); entry && count < MAX_GPUS; entry = strtok_r(NULL, ";", &save)) {
        Gpu *g = &gpus[count];
        unsigned long long total = 0, used = 0;
        char name[96] = "";
        if (sscanf(entry, "%95[^|]|%d.%d|%llu|%llu", name, &g->major, &g->minor, &total, &used) < 4) continue;
        snprintf(g->name, sizeof g->name, "%s", name);
        g->total = total << 20;
        g->used = used << 20;
        count++;
    }
    free(copy);
    return 0;
}

int nvmlInit_v2(void) { return load(); }
int nvmlInit(void) { return load(); }
int nvmlInitWithFlags(unsigned flags) { (void)flags; return load(); }
int nvmlShutdown(void) { return 0; }
const char *nvmlErrorString(int r) { return r == 9 ? "Driver Not Loaded" : "Unknown Error"; }

int nvmlSystemGetDriverVersion(char *buf, unsigned len) {
    const char *v = getenv("FAKE_NVML_DRIVER");
    snprintf(buf, len, "%s", v && *v ? v : "580.95.05");
    return 0;
}
int nvmlSystemGetCudaDriverVersion(int *v) { *v = getenv("FAKE_NVML_CUDA") ? atoi(getenv("FAKE_NVML_CUDA")) : 13000; return 0; }
int nvmlSystemGetCudaDriverVersion_v2(int *v) { return nvmlSystemGetCudaDriverVersion(v); }

int nvmlDeviceGetCount_v2(unsigned *n) { *n = count; return 0; }
int nvmlDeviceGetCount(unsigned *n) { *n = count; return 0; }
int nvmlDeviceGetHandleByIndex_v2(unsigned i, void **dev) {
    if (i >= count) return 2;
    *dev = &gpus[i];
    return 0;
}
int nvmlDeviceGetHandleByIndex(unsigned i, void **dev) { return nvmlDeviceGetHandleByIndex_v2(i, dev); }
int nvmlDeviceGetName(Gpu *g, char *buf, unsigned len) { snprintf(buf, len, "%s", g->name); return 0; }
int nvmlDeviceGetUUID(Gpu *g, char *buf, unsigned len) { snprintf(buf, len, "GPU-0000000%ld-0000-0000-0000-000000000000", (long)(g - gpus)); return 0; }
int nvmlDeviceGetCudaComputeCapability(Gpu *g, int *major, int *minor) { *major = g->major; *minor = g->minor; return 0; }
int nvmlDeviceGetMemoryInfo(Gpu *g, Memory *m) { m->total = g->total; m->used = g->used; m->free = g->total - g->used; return 0; }
int nvmlDeviceGetPciInfo_v3(Gpu *g, PciInfo *p) {
    long i = (long)(g - gpus);
    memset(p, 0, sizeof *p);
    p->bus = 1 + (unsigned)i;
    p->pciDeviceId = 0x2b8510de;
    snprintf(p->busId, sizeof p->busId, "00000000:%02x:00.0", p->bus);
    return 0;
}
int nvmlDeviceGetCurrPcieLinkGeneration(Gpu *g, unsigned *v) { (void)g; *v = 5; return 0; }
int nvmlDeviceGetMaxPcieLinkGeneration(Gpu *g, unsigned *v) { (void)g; *v = 5; return 0; }
int nvmlDeviceGetCurrPcieLinkWidth(Gpu *g, unsigned *v) { (void)g; *v = 16; return 0; }
int nvmlDeviceGetMaxPcieLinkWidth(Gpu *g, unsigned *v) { (void)g; *v = 16; return 0; }
int nvmlDeviceGetUtilizationRates(Gpu *g, Utilization *u) { (void)g; u->gpu = 0; u->memory = 0; return 0; }
int nvmlDeviceGetComputeRunningProcesses_v3(Gpu *g, unsigned *n, void *info) { (void)g; (void)info; *n = 0; return 0; }
int nvmlDeviceGetGraphicsRunningProcesses_v3(Gpu *g, unsigned *n, void *info) { (void)g; (void)info; *n = 0; return 0; }
int nvmlDeviceGetComputeRunningProcesses_v2(Gpu *g, unsigned *n, void *info) { return nvmlDeviceGetComputeRunningProcesses_v3(g, n, info); }
int nvmlDeviceGetGraphicsRunningProcesses_v2(Gpu *g, unsigned *n, void *info) { return nvmlDeviceGetGraphicsRunningProcesses_v3(g, n, info); }
