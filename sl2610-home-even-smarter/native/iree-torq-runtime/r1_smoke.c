// R1 spike: one native process, one IREE instance, two HAL devices (CPU local-task + Torq NPU).
// Links the Torq-fork IREE runtime statically (libiree_runtime_unified.a + driver_torq + TorqHw).
//
// usage: r1_smoke <cpu.vmfb> <torq.vmfb> [<encoder.vmfb> <encoder-func>] [--iree flags...]
//   cpu.vmfb  : module.main(f32[4], f32[4]) -> f32[4]  = (a+b)*a     on local-task
//   torq.vmfb : module.main(bf16[1,64], bf16[1,64]) -> bf16[1,64] = a+b on torq
//   encoder   : optional vendor encoder, one bf16[1,288,207] input, run on torq
// Flags like --torq_hw_type / --torq_device_allocator / --task_topology_* are parsed from argv.
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/api.h"
#include "iree/runtime/api.h"

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static uint16_t f32_to_bf16(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
static float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}

typedef struct {
  iree_hal_element_type_t type;
  iree_host_size_t rank;
  iree_hal_dim_t shape[4];
  float (*fill)(int input, size_t i);
} arg_spec_t;

static size_t spec_count(const arg_spec_t* s) {
  size_t n = 1; for (iree_host_size_t i = 0; i < s->rank; ++i) n *= s->shape[i]; return n;
}

// Builds inputs, invokes `func`, reads output 0 back to host and prints a summary.
static iree_status_t run_call(iree_runtime_session_t* session, const char* label,
                              const char* func, int n_inputs, const arg_spec_t* spec,
                              int iters) {
  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* dev_alloc = iree_runtime_session_device_allocator(session);
  iree_allocator_t host_alloc = iree_runtime_session_host_allocator(session);
  size_t n = spec_count(spec);
  size_t esz = iree_hal_element_dense_byte_count(spec->type);
  uint8_t* host = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_alloc, n * esz, (void**)&host));

  iree_runtime_call_t call;
  iree_status_t status = iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view(func), &call);
  for (int in = 0; in < n_inputs && iree_status_is_ok(status); ++in) {
    for (size_t i = 0; i < n; ++i) {
      float v = spec->fill(in, i);
      if (spec->type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16) ((uint16_t*)host)[i] = f32_to_bf16(v);
      else ((float*)host)[i] = v;
    }
    iree_hal_buffer_view_t* view = NULL;
    status = iree_hal_buffer_view_allocate_buffer_copy(
        device, dev_alloc, spec->rank, spec->shape, spec->type,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                                   .access = IREE_HAL_MEMORY_ACCESS_ALL,
                                   .usage = IREE_HAL_BUFFER_USAGE_DEFAULT},
        iree_make_const_byte_span(host, n * esz), &view);
    if (iree_status_is_ok(status)) status = iree_runtime_call_inputs_push_back_buffer_view(&call, view);
    iree_hal_buffer_view_release(view);
  }

  double best = 1e30, first = 0;
  for (int it = 0; it < iters && iree_status_is_ok(status); ++it) {
    iree_vm_list_resize(iree_runtime_call_outputs(&call), 0);
    double t0 = now_ms();
    status = iree_runtime_call_invoke(&call, 0);
    double dt = now_ms() - t0;
    if (it == 0) first = dt;
    if (dt < best) best = dt;
  }

  iree_hal_buffer_view_t* out = NULL;
  if (iree_status_is_ok(status)) status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &out);
  if (iree_status_is_ok(status)) {
    iree_device_size_t bytes = iree_hal_buffer_view_byte_length(out);
    iree_hal_element_type_t ot = iree_hal_buffer_view_element_type(out);
    size_t oesz = iree_hal_element_dense_byte_count(ot);
    size_t on = (size_t)bytes / oesz;
    uint8_t* ob = NULL;
    status = iree_allocator_malloc(host_alloc, bytes, (void**)&ob);
    if (iree_status_is_ok(status))
      status = iree_hal_device_transfer_d2h(device, iree_hal_buffer_view_buffer(out), 0, ob, bytes,
                                            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    if (iree_status_is_ok(status)) {
      size_t nz = 0; double absum = 0;
      printf("[%s] %s out=%zu elems:", label, func, on);
      for (size_t i = 0; i < on; ++i) {
        float v = ot == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 ? bf16_to_f32(((uint16_t*)ob)[i]) : ((float*)ob)[i];
        if (v != 0.f) ++nz;
        absum += fabs(v);
        if (i < 6) printf(" %g", v);
      }
      printf(" | nonzero=%.3f absmean=%.4f | first=%.2fms best=%.2fms (%d iters)\n",
             on ? (double)nz / on : 0, on ? absum / on : 0, first, best, iters);
    }
    iree_allocator_free(host_alloc, ob);
    iree_hal_buffer_view_release(out);
  }
  iree_runtime_call_deinitialize(&call);
  iree_allocator_free(host_alloc, host);
  return status;
}

static float fill_cpu(int in, size_t i) { return in == 0 ? (float)(i + 1) : 10.f * (i + 1); }
static float fill_npu(int in, size_t i) { return in == 0 ? (float)(i % 8) : 0.5f; }
static float fill_enc(int in, size_t i) { (void)in; return (float)((i * 2654435761u) % 1000) / 1000.f - 0.5f; }

static iree_status_t make_session(iree_runtime_instance_t* instance, const char* uri,
                                  const char* vmfb, iree_runtime_session_t** out) {
  iree_hal_device_t* device = NULL;
  double t0 = now_ms();
  IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view(uri), &device));
  iree_runtime_session_options_t opts;
  iree_runtime_session_options_initialize(&opts);
  iree_status_t status = iree_runtime_session_create_with_device(
      instance, &opts, device, iree_runtime_instance_host_allocator(instance), out);
  iree_hal_device_release(device);
  if (iree_status_is_ok(status))
    status = iree_runtime_session_append_bytecode_module_from_file(*out, vmfb);
  printf("[%s] device+session+module(%s) in %.1f ms\n", uri, vmfb, now_ms() - t0);
  return status;
}

static iree_status_t run(int argc, char** argv) {
  iree_runtime_instance_options_t io;
  iree_runtime_instance_options_initialize(&io);
  iree_runtime_instance_options_use_all_available_drivers(&io);
  iree_runtime_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_create(&io, iree_allocator_system(), &instance));

  iree_host_size_t n = 0;
  iree_hal_driver_info_t* infos = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_enumerate(
      iree_runtime_instance_driver_registry(instance), iree_allocator_system(), &n, &infos));
  printf("drivers:");
  for (iree_host_size_t i = 0; i < n; ++i)
    printf(" %.*s", (int)infos[i].driver_name.size, infos[i].driver_name.data);
  printf("\n");
  iree_allocator_free(iree_allocator_system(), infos);

  iree_runtime_session_t *cpu = NULL, *npu = NULL;
  iree_status_t s = make_session(instance, "local-task", argv[1], &cpu);
  if (iree_status_is_ok(s)) s = make_session(instance, "torq", argv[2], &npu);

  arg_spec_t cpu_spec = {IREE_HAL_ELEMENT_TYPE_FLOAT_32, 1, {4}, fill_cpu};
  arg_spec_t npu_spec = {IREE_HAL_ELEMENT_TYPE_BFLOAT_16, 2, {1, 64}, fill_npu};
  // Interleave CPU and NPU calls in the same process to show both devices coexist.
  for (int round = 0; round < 2 && iree_status_is_ok(s); ++round) {
    s = run_call(cpu, "local-task", "module.main", 2, &cpu_spec, 20);
    if (iree_status_is_ok(s)) s = run_call(npu, "torq", "module.main", 2, &npu_spec, 20);
  }
  if (iree_status_is_ok(s) && argc >= 5 && argv[3][0] != '-') {
    iree_runtime_session_t* enc = NULL;
    s = make_session(instance, "torq", argv[3], &enc);
    arg_spec_t enc_spec = {IREE_HAL_ELEMENT_TYPE_BFLOAT_16, 3, {1, 288, 207}, fill_enc};
    if (iree_status_is_ok(s)) s = run_call(enc, "torq-encoder", argv[4], 1, &enc_spec, 5);
    if (iree_status_is_ok(s)) s = run_call(cpu, "local-task", "module.main", 2, &cpu_spec, 5);
    iree_runtime_session_release(enc);
  }
  iree_runtime_session_release(npu);
  iree_runtime_session_release(cpu);
  iree_runtime_instance_release(instance);
  return s;
}

int main(int argc, char** argv) {
  // Consume --flags (torq_hw_type, torq_device_allocator, task_topology_*); keep positionals.
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc < 3) {
    fprintf(stderr, "usage: %s <cpu.vmfb> <torq.vmfb> [<encoder.vmfb> <func>] [--flags]\n", argv[0]);
    return 2;
  }
  iree_status_t s = run(argc, argv);
  if (!iree_status_is_ok(s)) {
    iree_status_fprint(stderr, s);
    iree_status_free(s);
    return 1;
  }
  printf("R1 OK\n");
  return 0;
}
