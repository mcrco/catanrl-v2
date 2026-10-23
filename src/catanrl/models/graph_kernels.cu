// Inference kernels for catanrl's shared-trunk graph network (models/graph.py), used by
// models/graph_kernels.py. Matrix products stay in cuBLAS (torch.mm / addmm); these kernels
// do the rest: the topology as neighbour lists, residual + LayerNorm, neighbour means,
// pooling and the output assembly (the one- and two-output head layers as warp dot
// products). A row's work reads only that row, so its outputs never depend on the other
// rows of its batch. Kernel structs are selected by value per blockIdx.y: taking a
// parameter's address puts it in local memory, which cost 30x on these kernels.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace {

constexpr int MAXD = 3;   // node degree caps (hex boards: <= 3 neighbours, edges, tiles)
constexpr int CORNERS = 6;
constexpr int HPL = 3;    // hidden / 32 (hidden 96)
constexpr int H = 32 * HPL;

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

struct Layout {
  int B, R, N, T, E;
  int off_nodes, off_tiles, off_edges, wn, wt, we;  // raw blocks
  int vn, vt, ve;                                    // valid feature (raw index)
  int off_en, off_tn;                                // edge_nodes, tile_nodes
};

// One block per row: valid flags, the raw entity features copied contiguous, and the
// neighbour lists in the dense incidence order (models/graph.py topology()).
__global__ void topo_kernel(const float* __restrict__ obs, Layout d,
                            float* xn, float* xt, float* xe, float* vn, float* vt, float* ve,
                            int* nn, int* nnc, int* ne, int* nec, int* nt, int* ntc,
                            int* tn, int* tnc, int* en, int* enc, int* flag) {
  const int b = blockIdx.x;
  const float* row = obs + (long)b * d.R;
  const float* EN = row + d.off_en;
  const float* TN = row + d.off_tn;
  for (int k = threadIdx.x; k < d.N * d.wn; k += blockDim.x) xn[(long)b * d.N * d.wn + k] = row[d.off_nodes + k];
  for (int k = threadIdx.x; k < d.T * d.wt; k += blockDim.x) xt[(long)b * d.T * d.wt + k] = row[d.off_tiles + k];
  for (int k = threadIdx.x; k < d.E * d.we; k += blockDim.x) xe[(long)b * d.E * d.we + k] = row[d.off_edges + k];
  for (int i = threadIdx.x; i < d.N; i += blockDim.x) {
    const long g = (long)b * d.N + i;
    vn[g] = row[d.off_nodes + i * d.wn + d.vn];
    int cn = 0, ce = 0, ct = 0;
    for (int e = 0; e < d.E; ++e) {
      const int a = (int)rintf(EN[2 * e]), c = (int)rintf(EN[2 * e + 1]);
      if (a == i) { if (ce < MAXD) ne[g * MAXD + ce] = e; ++ce; }
      if (c == i) { if (ce < MAXD) ne[g * MAXD + ce] = e; ++ce; }
      if (a >= 0 && a < d.N && c >= 0 && c < d.N) {  // adjacency = first^T second + its transpose
        if (a == i) { if (cn < MAXD) nn[g * MAXD + cn] = c; ++cn; }
        if (c == i) { if (cn < MAXD) nn[g * MAXD + cn] = a; ++cn; }
      }
    }
    for (int t = 0; t < d.T; ++t)
      for (int k = 0; k < CORNERS; ++k)
        if ((int)rintf(TN[t * CORNERS + k]) == i) { if (ct < MAXD) nt[g * MAXD + ct] = t; ++ct; }
    nnc[g] = cn; nec[g] = ce; ntc[g] = ct;
    if (cn > MAXD || ce > MAXD || ct > MAXD) atomicExch(flag, 1);
  }
  for (int t = threadIdx.x; t < d.T; t += blockDim.x) {
    const long g = (long)b * d.T + t;
    vt[g] = row[d.off_tiles + t * d.wt + d.vt];
    int c = 0;
    for (int k = 0; k < CORNERS; ++k) {
      const int v = (int)rintf(TN[t * CORNERS + k]);
      if (v >= 0 && v < d.N) tn[g * CORNERS + c++] = v;
    }
    tnc[g] = c;
  }
  for (int e = threadIdx.x; e < d.E; e += blockDim.x) {
    const long g = (long)b * d.E + e;
    ve[g] = row[d.off_edges + e * d.we + d.ve];
    int c = 0;
    for (int k = 0; k < 2; ++k) {
      const int v = (int)rintf(EN[2 * e + k]);
      if (v >= 0 && v < d.N) en[g * 2 + c++] = v;
    }
    enc[g] = c;
  }
}

struct Entity {
  float* x;            // [count, H] the entity's embedding (updated in place)
  const float* u;      // [count, H] update from the GEMM, or null
  const float* ub;     // [H] the update's bias, or null
  const float* gb;     // injection: [B, gbs] glob part (bias in), or null for a residual step
  int gbs;
  const float* inj_g; const float* inj_b;
  const float* valid;  // [count]
  float* z; int zs;    // LayerNorm output (row stride zs), or null
  const float* g; const float* bt;
  float eps, inj_eps;
  int count; int per_row;
};

__device__ __forceinline__ void layer_norm(float v[HPL], const float* g, const float* bt, float eps, int lane) {
  float s = 0.f;
  for (int k = 0; k < HPL; ++k) s += v[k];
  const float mean = warp_sum(s) / H;
  float q = 0.f;
  for (int k = 0; k < HPL; ++k) { const float dd = v[k] - mean; q += dd * dd; }
  const float rstd = rsqrtf(warp_sum(q) / H + eps);
  for (int k = 0; k < HPL; ++k) { const int h = lane + 32 * k; v[k] = (v[k] - mean) * rstd * g[h] + bt[h]; }
}

// One warp per entity of the three types: injection (x = relu(LN(u + gb)) * valid) or a
// residual step (x = (x + relu(u)) * valid, u optional), then z = LN(x) when z is set.
__global__ void entity_kernel(Entity a0, Entity a1, Entity a2) {
  // blockIdx.y picks the entity type: the selected struct lives in registers.
  const Entity a = blockIdx.y == 0 ? a0 : blockIdx.y == 1 ? a1 : a2;
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (w >= a.count) return;
  float v[HPL];
  const float m = a.valid[w];
  if (a.gb) {
    const float* gb = a.gb + (w / a.per_row) * a.gbs;
    for (int k = 0; k < HPL; ++k) { const int h = lane + 32 * k; v[k] = a.u[w * H + h] + gb[h]; }
    layer_norm(v, a.inj_g, a.inj_b, a.inj_eps, lane);
    for (int k = 0; k < HPL; ++k) v[k] = fmaxf(v[k], 0.f) * m;
  } else {
    for (int k = 0; k < HPL; ++k) {
      const int h = lane + 32 * k;
      v[k] = a.x[w * H + h];
      if (a.u) v[k] = (v[k] + fmaxf(a.u[w * H + h] + (a.ub ? a.ub[h] : 0.f), 0.f)) * m;
    }
  }
  if (a.gb || a.u)
    for (int k = 0; k < HPL; ++k) a.x[w * H + lane + 32 * k] = v[k];
  if (a.z) {
    layer_norm(v, a.g, a.bt, a.eps, lane);
    for (int k = 0; k < HPL; ++k) a.z[w * a.zs + lane + 32 * k] = v[k];
  }
}

struct Gather {
  const float* src; int ss; int src_per_row;
  float* dst; int ds; int dst_per_row;
  const int* idx; const int* cnt; int maxd;
  int count;
  int mean;  // 1: mean over the list (clamped count), 0: sum
};

// One warp per (entity, aggregation): dst = mean or sum of src rows in the entity's list;
// blockIdx.y picks the aggregation.
__global__ void gather_kernel(Gather g0, Gather g1, Gather g2, Gather g3, Gather g4) {
  const int y = blockIdx.y;
  const Gather g = y == 0 ? g0 : y == 1 ? g1 : y == 2 ? g2 : y == 3 ? g3 : g4;
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (w >= g.count) return;
  const int b = w / g.dst_per_row;
  const int c = g.cnt[w];
  float acc[HPL] = {0.f, 0.f, 0.f};
  for (int j = 0; j < c && j < g.maxd; ++j) {
    const float* s = g.src + (b * g.src_per_row + g.idx[w * g.maxd + j]) * g.ss;
    for (int k = 0; k < HPL; ++k) acc[k] += s[lane + 32 * k];
  }
  const float scale = g.mean ? 1.f / fmaxf((float)c, 1.f) : 1.f;
  for (int k = 0; k < HPL; ++k) g.dst[w * g.ds + lane + 32 * k] = acc[k] * scale;
}

// One block per row: [masked mean of nodes | tiles | edges | glob].
__global__ void pool_kernel(const float* xn, const float* xt, const float* xe, const float* vn, const float* vt,
                            const float* ve, const float* glob, int N, int T, int E, int Hg, float* pooled) {
  const int b = blockIdx.x;
  const int P = 3 * H + Hg;
  for (int j = threadIdx.x; j < P; j += blockDim.x) {
    float out;
    if (j < 3 * H) {
      const int type = j / H, h = j % H;
      const float* x = type == 0 ? xn : type == 1 ? xt : xe;
      const float* v = type == 0 ? vn : type == 1 ? vt : ve;
      const int n = type == 0 ? N : type == 1 ? T : E;
      float s = 0.f, c = 0.f;
      for (int i = 0; i < n; ++i) { const float m = v[(long)b * n + i]; s += x[((long)b * n + i) * H + h] * m; c += m; }
      out = s / fmaxf(c, 1.f);
    } else {
      out = glob[(long)b * Hg + (j - 3 * H)];
    }
    pooled[(long)b * P + j] = out;
  }
}

struct Assemble {
  const float* obs; int R, off_mask, A, G, N, T, E, RS, Hh, P;
  const float* other;   // [B, G]
  const float* sch; int Hs;  // [B*N, 2*Hs] settlement | city hidden (relu'd)
  const float* ws2; const float* wc2; const float* bsc;  // [Hs], [Hs], [2]
  const float* roadh; int Hr; const float* wr3; const float* br3;  // [B*E, Hr] road hidden (relu'd)
  const float* rt;      // [B*T, Hh] robber_tile(tiles) (bias in)
  const float* rv;      // [B, RS, Hh] robber_victim(victims)
  const float* wro; const float* bro;
  const float* vl;      // [B*P] value MLP output
  int off_players, wp, present;
  float* logits; float* values;
};

__device__ __forceinline__ float warp_dot(const float* a, const float* w, int n, int lane) {
  float acc = 0.f;
  for (int h = lane; h < n; h += 32) acc += a[h] * w[h];
  return warp_sum(acc);
}

// One block per row, one warp per logit: the dot-product heads (settlement, city, road,
// robber) split their hidden units over the warp's lanes.
__global__ void assemble_kernel(Assemble s) {
  const int b = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  const float* row = s.obs + (long)b * s.R;
  for (int j = warp; j < s.A; j += warps) {
    float v;
    int k = j;
    if (k < s.G) v = s.other[(long)b * s.G + k];
    else if ((k -= s.G) < 2 * s.N) {  // settlement then city
      const int city = k >= s.N, i = city ? k - s.N : k;
      const float* hrow = s.sch + ((long)b * s.N + i) * 2 * s.Hs + (city ? s.Hs : 0);
      v = warp_dot(hrow, city ? s.wc2 : s.ws2, s.Hs, lane) + s.bsc[city];
    } else if ((k -= 2 * s.N) < s.E) {
      v = warp_dot(s.roadh + ((long)b * s.E + k) * s.Hr, s.wr3, s.Hr, lane) + s.br3[0];
    } else if ((k -= s.E) < s.T * s.RS) {
      const int t = k / s.RS, p = k % s.RS;
      const float* rt = s.rt + ((long)b * s.T + t) * s.Hh;
      const float* rv = s.rv + ((long)b * s.RS + p) * s.Hh;
      float acc = 0.f;
      for (int h = lane; h < s.Hh; h += 32) acc += fmaxf(rt[h] + rv[h], 0.f) * s.wro[h];
      v = warp_sum(acc) + s.bro[0];
    } else v = 0.f;  // the no-op
    if (lane == 0) s.logits[(long)b * s.A + j] = row[s.off_mask + j] > 0.f ? v : -1e9f;
  }
  if (threadIdx.x == 0) {
    float n = 0.f;
    for (int p = 0; p < s.P; ++p) n += row[s.off_players + p * s.wp + s.present];
    const float prior = -logf(fmaxf(n, 2.f) - 1.f);
    for (int p = 0; p < s.P; ++p) {
      const float present = row[s.off_players + p * s.wp + s.present];
      s.values[(long)b * s.P + p] = present / (1.f + expf(-(s.vl[(long)b * s.P + p] + prior)));
    }
  }
}


const float* ptr(const c10::optional<torch::Tensor>& t) { return t ? t->data_ptr<float>() : nullptr; }

}  // namespace

std::vector<torch::Tensor> topo(torch::Tensor obs, std::vector<int64_t> l) {
  Layout d{(int)obs.size(0), (int)obs.size(1), (int)l[0], (int)l[1], (int)l[2], (int)l[3], (int)l[4], (int)l[5],
           (int)l[6], (int)l[7], (int)l[8], (int)l[9], (int)l[10], (int)l[11], (int)l[12], (int)l[13]};
  auto f = obs.options();
  auto i = obs.options().dtype(torch::kInt32);
  auto xn = torch::empty({d.B * d.N, d.wn}, f), xt = torch::empty({d.B * d.T, d.wt}, f), xe = torch::empty({d.B * d.E, d.we}, f);
  auto vn = torch::empty({d.B * d.N}, f), vt = torch::empty({d.B * d.T}, f), ve = torch::empty({d.B * d.E}, f);
  auto nn = torch::zeros({d.B * d.N, MAXD}, i), ne = torch::zeros({d.B * d.N, MAXD}, i), nt = torch::zeros({d.B * d.N, MAXD}, i);
  auto nnc = torch::empty({d.B * d.N}, i), nec = torch::empty({d.B * d.N}, i), ntc = torch::empty({d.B * d.N}, i);
  auto tn = torch::zeros({d.B * d.T, CORNERS}, i), tnc = torch::empty({d.B * d.T}, i);
  auto en = torch::zeros({d.B * d.E, 2}, i), enc = torch::empty({d.B * d.E}, i);
  auto flag = torch::zeros({1}, i);
  topo_kernel<<<d.B, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      obs.data_ptr<float>(), d, xn.data_ptr<float>(), xt.data_ptr<float>(), xe.data_ptr<float>(),
      vn.data_ptr<float>(), vt.data_ptr<float>(), ve.data_ptr<float>(), nn.data_ptr<int>(), nnc.data_ptr<int>(),
      ne.data_ptr<int>(), nec.data_ptr<int>(), nt.data_ptr<int>(), ntc.data_ptr<int>(), tn.data_ptr<int>(),
      tnc.data_ptr<int>(), en.data_ptr<int>(), enc.data_ptr<int>(), flag.data_ptr<int>());
  return {xn, xt, xe, vn, vt, ve, nn, nnc, ne, nec, nt, ntc, tn, tnc, en, enc, flag};
}

// Per type t in 0..2: x, u?, ub?, gb?, gbs, inj_g?, inj_b?, inj_eps, valid, z?, zs, g?, bt?, eps, per_row.
void entities(std::vector<c10::optional<torch::Tensor>> x, std::vector<c10::optional<torch::Tensor>> u,
              std::vector<c10::optional<torch::Tensor>> ub,
              std::vector<c10::optional<torch::Tensor>> gb, std::vector<int64_t> gbs,
              std::vector<c10::optional<torch::Tensor>> inj_g, std::vector<c10::optional<torch::Tensor>> inj_b,
              std::vector<double> inj_eps, std::vector<torch::Tensor> valid,
              std::vector<c10::optional<torch::Tensor>> z, std::vector<int64_t> zs,
              std::vector<c10::optional<torch::Tensor>> g, std::vector<c10::optional<torch::Tensor>> bt,
              std::vector<double> eps, std::vector<int64_t> per_row) {
  Entity a[3];
  int most = 0;
  for (int t = 0; t < 3; ++t) {
    a[t] = Entity{x[t]->data_ptr<float>(), ptr(u[t]), ptr(ub[t]), ptr(gb[t]), (int)gbs[t], ptr(inj_g[t]), ptr(inj_b[t]),
                  valid[t].data_ptr<float>(), z[t] ? z[t]->data_ptr<float>() : nullptr, (int)zs[t], ptr(g[t]),
                  ptr(bt[t]), (float)eps[t], (float)inj_eps[t], (int)valid[t].numel(), (int)per_row[t]};
    most = std::max(most, a[t].count);
  }
  const int threads = 256;
  const dim3 blocks((most * 32 + threads - 1) / threads, 3);
  entity_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(a[0], a[1], a[2]);
}

// Each gather: src, ss, src_per_row, dst, ds, dst_per_row, idx, cnt, mean.
void gathers(std::vector<torch::Tensor> src, std::vector<int64_t> ss, std::vector<int64_t> spr,
             std::vector<torch::Tensor> dst, std::vector<int64_t> ds, std::vector<int64_t> dpr,
             std::vector<torch::Tensor> idx, std::vector<torch::Tensor> cnt, std::vector<int64_t> mean) {
  Gather g[5] = {};
  const int n = (int)src.size();
  int most = 0;
  for (int i = 0; i < n; ++i) {
    g[i] = Gather{src[i].data_ptr<float>(), (int)ss[i], (int)spr[i], dst[i].data_ptr<float>(), (int)ds[i],
                  (int)dpr[i], idx[i].data_ptr<int>(), cnt[i].data_ptr<int>(), (int)idx[i].size(1),
                  (int)cnt[i].numel(), (int)mean[i]};
    most = std::max(most, g[i].count);
  }
  const int threads = 256;
  const dim3 blocks((most * 32 + threads - 1) / threads, n);
  gather_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(g[0], g[1], g[2], g[3], g[4]);
}

torch::Tensor pool(torch::Tensor xn, torch::Tensor xt, torch::Tensor xe, torch::Tensor vn, torch::Tensor vt,
                   torch::Tensor ve, torch::Tensor glob, int64_t N, int64_t T, int64_t E) {
  const int B = glob.size(0), Hg = glob.size(1);
  auto pooled = torch::empty({B, 3 * H + Hg}, glob.options());
  pool_kernel<<<B, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      xn.data_ptr<float>(), xt.data_ptr<float>(), xe.data_ptr<float>(), vn.data_ptr<float>(), vt.data_ptr<float>(),
      ve.data_ptr<float>(), glob.data_ptr<float>(), (int)N, (int)T, (int)E, Hg, pooled.data_ptr<float>());
  return pooled;
}

std::vector<torch::Tensor> assemble(torch::Tensor obs, std::vector<int64_t> l, torch::Tensor other, torch::Tensor sch,
                                    torch::Tensor ws2, torch::Tensor wc2, torch::Tensor bsc, torch::Tensor roadh,
                                    torch::Tensor wr3, torch::Tensor br3, torch::Tensor rt, torch::Tensor rv, torch::Tensor wro,
                                    torch::Tensor bro, torch::Tensor vl) {
  // l: off_mask, A, G, N, T, E, RS, Hh, P, off_players, wp, present
  const int B = obs.size(0);
  auto logits = torch::empty({B, l[1]}, obs.options());
  auto values = torch::empty({B, l[8]}, obs.options());
  Assemble s{obs.data_ptr<float>(), (int)obs.size(1), (int)l[0], (int)l[1], (int)l[2], (int)l[3], (int)l[4],
             (int)l[5], (int)l[6], (int)l[7], (int)l[8], other.data_ptr<float>(), sch.data_ptr<float>(),
             (int)(sch.size(1) / 2), ws2.data_ptr<float>(), wc2.data_ptr<float>(), bsc.data_ptr<float>(),
             roadh.data_ptr<float>(), (int)roadh.size(1), wr3.data_ptr<float>(), br3.data_ptr<float>(), rt.data_ptr<float>(), rv.data_ptr<float>(), wro.data_ptr<float>(),
             bro.data_ptr<float>(), vl.data_ptr<float>(), (int)l[9], (int)l[10], (int)l[11],
             logits.data_ptr<float>(), values.data_ptr<float>()};
  assemble_kernel<<<B, 512, 0, at::cuda::getCurrentCUDAStream()>>>(s);
  return {logits, values};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topo", &topo);
  m.def("entities", &entities);
  m.def("gathers", &gathers);
  m.def("pool", &pool);
  m.def("assemble", &assemble);
  m.attr("hidden") = H;
}
