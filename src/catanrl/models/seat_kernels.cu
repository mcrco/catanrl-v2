// Inference kernels for catanrl's seat-set graph network (models/seat.py), used by
// models/seat_kernels.py beside the graph network's (graph_kernels.cu: the topology, the
// board entities' residual + LayerNorm, their neighbour means, the pooling and the output
// assembly are those). These do what the seats add: the seats' incidence with the board and
// the slot-free board inputs, the seat tokens, a seat's messages summed into its nodes, edges
// and tiles, the board's weighted means and the seat-to-seat mixing back into the seats, and
// the tokens' residual + LayerNorm. Matrix products stay in cuBLAS. A row's work reads only
// that row. Kernel structs are passed by value (graph_kernels.cu says why).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int HPL = 3;          // board hidden / 32 (hidden 96, graph_kernels.cu's)
constexpr int H = 32 * HPL;
constexpr int MAXP = 6;         // seat slots (catanrl.envs.encoding.MAX_PLAYERS)
constexpr int NR = 5;           // seat <-> node relations (seat.NODE_RELATIONS)
constexpr int GROUPS = NR + 2;  // a seat's messages: the node relations', the edges', the tiles'
constexpr int SPL = 8;          // seat_hidden <= 32 * SPL
constexpr int NONFINITE = 2;    // the flag's bit for a half that is not finite (graph_kernels.cu)
constexpr float HALF_MAX = 65504.f;

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float col(const float* x, int c) { return c >= 0 ? x[c] : 0.f; }

__device__ __forceinline__ float load(const float* p) { return *p; }
__device__ __forceinline__ float load(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void store(float* p, float v, bool&) { *p = v; }
__device__ __forceinline__ void store(__half* p, float v, bool& bad) {
  bad |= !(fabsf(v) <= HALF_MAX);  // inf and nan
  *p = __float2half(v);
}

struct Seats {
  int B, R, N, T, E, P, S;
  int off_global, off_players, off_nodes, off_tiles, off_edges;  // raw blocks
  int wp, wn, wt, we;                                            // raw widths
  int vn, vt, ve, present;                                       // valid / present feature (raw index)
  // Raw columns by feature (-1: a feature the rows lack, read as 0), offsets into the table:
  int kn, node_plain, settlement, city, distance;                // kn plain features, then P columns each
  int kt, tile_plain, buildings, ktd, tile_odds, tile_dice;      // ktd < 0: rows without the dice input
  int ke, edge_plain, road;
  int kp, player;                                                // a seat's features
  int kg, global_plain, kd, global_dice, ks, seat_dice;          // seat_dice: [P, ks]
  float on, one, near;                                           // the distance bands' edges
};

// One block per row (seat.py relations() and the first projections' inputs): each entity's
// plain features, its "mine" and "another seat's" channels and its weight by seat and relation;
// the seats' features; the global projection's plain and dice inputs (the tokens' summary
// between them is tokens_kernel's); each seat's summed weights by relation (at least 1e-6).
__global__ void relations_kernel(const float* __restrict__ obs, const int* __restrict__ tab, Seats d,
                                 float* xn, float* xt, float* xe, float* xs, float* gin, float* present,
                                 float* wN, float* wE, float* wT, float* den) {
  const int b = blockIdx.x;
  const float* row = obs + (long)b * d.R;
  float seat[MAXP];
  float total = 0.f;
  for (int p = 0; p < d.P; ++p) { seat[p] = row[d.off_players + p * d.wp + d.present]; total += seat[p]; }
  const float others = fmaxf(total - seat[0], 1.f);
  if (threadIdx.x < d.P) present[(long)b * d.P + threadIdx.x] = seat[threadIdx.x];
  for (int i = threadIdx.x; i < d.N; i += blockDim.x) {
    const float* x = row + d.off_nodes + i * d.wn;
    const long g = (long)b * d.N + i;
    float* out = xn + g * (d.kn + 6);
    float* w = wN + g * NR * d.P;
    for (int k = 0; k < d.kn; ++k) out[k] = col(x, tab[d.node_plain + k]);
    const float valid = x[d.vn];
    float settlements = 0.f, cities = 0.f, nearest = 1.f;
    for (int p = 0; p < d.P; ++p) {
      const float s = col(x, tab[d.settlement + p]), c = col(x, tab[d.city + p]);
      const float dist = col(x, tab[d.distance + p]);
      if (p > 0) {
        settlements += s * seat[p];
        cities += c * seat[p];
        if (seat[p] > 0.f) nearest = fminf(nearest, dist);
      }
      const float m = valid * seat[p];
      w[p] = s * m;
      w[d.P + p] = c * m;
      w[2 * d.P + p] = (dist < d.on ? 1.f : 0.f) * m;
      w[3 * d.P + p] = (dist >= d.on && dist < d.one ? 1.f : 0.f) * m;
      w[4 * d.P + p] = (dist >= d.one && dist < d.near ? 1.f - dist : 0.f) * m;
    }
    out += d.kn;
    out[0] = col(x, tab[d.settlement]); out[1] = settlements;
    out[2] = col(x, tab[d.city]); out[3] = cities;
    out[4] = col(x, tab[d.distance]); out[5] = nearest;
  }
  for (int i = threadIdx.x; i < d.T; i += blockDim.x) {
    const float* x = row + d.off_tiles + i * d.wt;
    const long g = (long)b * d.T + i;
    float* out = xt + g * (d.kt + 2 + (d.ktd >= 0 ? 2 + d.ktd : 0));
    for (int k = 0; k < d.kt; ++k) out[k] = col(x, tab[d.tile_plain + k]);
    const float valid = x[d.vt];
    float buildings = 0.f, odds = 0.f;
    for (int p = 0; p < d.P; ++p) {
      const float v = col(x, tab[d.buildings + p]);
      if (p > 0) {
        buildings += v * seat[p];
        if (d.ktd >= 0) odds += col(x, tab[d.tile_odds + p]) * seat[p];
      }
      wT[g * d.P + p] = v * 6.f * valid * seat[p];
    }
    out += d.kt;
    out[0] = col(x, tab[d.buildings]); out[1] = buildings;
    if (d.ktd >= 0) {
      out[2] = col(x, tab[d.tile_odds]); out[3] = odds / others;
      for (int k = 0; k < d.ktd; ++k) out[4 + k] = col(x, tab[d.tile_dice + k]);
    }
  }
  for (int i = threadIdx.x; i < d.E; i += blockDim.x) {
    const float* x = row + d.off_edges + i * d.we;
    const long g = (long)b * d.E + i;
    float* out = xe + g * (d.ke + 2);
    for (int k = 0; k < d.ke; ++k) out[k] = col(x, tab[d.edge_plain + k]);
    const float valid = x[d.ve];
    float roads = 0.f;
    for (int p = 0; p < d.P; ++p) {
      const float v = col(x, tab[d.road + p]);
      if (p > 0) roads += v * seat[p];
      wE[g * d.P + p] = v * valid * seat[p];
    }
    out[d.ke] = col(x, tab[d.road]); out[d.ke + 1] = roads;
  }
  const int ws = d.kp + d.ks;
  for (int j = threadIdx.x; j < d.P * ws; j += blockDim.x) {
    const int p = j / ws, k = j % ws;
    xs[(long)b * d.P * ws + j] = k < d.kp ? col(row + d.off_players + p * d.wp, tab[d.player + k])
                                           : col(row + d.off_global, tab[d.seat_dice + p * d.ks + k - d.kp]);
  }
  float* g = gin + (long)b * (d.kg + 3 * d.S + d.kd);
  for (int j = threadIdx.x; j < d.kg; j += blockDim.x) g[j] = col(row + d.off_global, tab[d.global_plain + j]);
  for (int j = threadIdx.x; j < d.kd; j += blockDim.x)
    g[d.kg + 3 * d.S + j] = col(row + d.off_global, tab[d.global_dice + j]);
  __syncthreads();  // the weights above, summed over each seat's entities
  for (int j = threadIdx.x; j < GROUPS * d.P; j += blockDim.x) {
    const int group = j / d.P, p = j % d.P;
    float sum = 0.f;
    if (group < NR) for (int i = 0; i < d.N; ++i) sum += wN[((long)b * d.N + i) * NR * d.P + j];
    else if (group == NR) for (int i = 0; i < d.E; ++i) sum += wE[((long)b * d.E + i) * d.P + p];
    else for (int i = 0; i < d.T; ++i) sum += wT[((long)b * d.T + i) * d.P + p];
    den[(long)b * GROUPS * d.P + j] = fmaxf(sum, 1e-6f);
  }
}

// The seats present of a row, as the module counts them (seat.py relations()).
__device__ __forceinline__ int seats_present(const float* present, int P, float& total) {
  total = 0.f;
  for (int p = 0; p < P; ++p) total += present[p];
  return min(max((int)rintf(total), 0), MAXP);
}

// One block per row: the tokens before the injection, relu(projection + the turn-order
// features' term, bias in) * present, and the global projection's summary of them: the row's
// own token, the mean and the maximum of the other seats'.
__global__ void tokens_kernel(const float* __restrict__ ps, const float* __restrict__ turn,
                              const float* __restrict__ present, int P, int S, float* seats, float* gin, int gw,
                              int offset) {
  const int b = blockIdx.x;
  const float* here = present + (long)b * P;
  float total;
  const int count = seats_present(here, P, total);
  const float others = fmaxf(total - here[0], 1.f);
  const float* tb = turn + (long)count * P * S;
  for (int h = threadIdx.x; h < S; h += blockDim.x) {
    float own = 0.f, sum = 0.f, most = 0.f;
    for (int p = 0; p < P; ++p) {
      const long j = ((long)b * P + p) * S + h;
      const float v = fmaxf(ps[j] + tb[p * S + h], 0.f) * here[p];
      seats[j] = v;
      if (p == 0) own = v; else { sum += v; most = fmaxf(most, v); }
    }
    float* g = gin + (long)b * gw + offset;
    g[h] = own; g[S + h] = sum / others; g[2 * S + h] = most;
  }
}

struct Step {
  float* x;                  // [count, S] the tokens (updated in place)
  const float* u; int us;    // [count, us]: the update's first term in its first S columns, or null
  const float* u2;           // [count, S] its second term (with u)
  const float* ub;           // [S] its bias (with u)
  const float* gb; int gbs;  // injection: [B, gbs] glob part (bias in), or null for a residual step
  const float* inj_g; const float* inj_b; float inj_eps;
  const float* present;      // [count]
  float* z;                  // [count, S] LayerNorm output, or null
  const float* g; const float* bt; float eps;
  int count, S, P;
};

__device__ __forceinline__ void seat_norm(float v[SPL], int n, int S, const float* g, const float* bt, float eps,
                                          int lane) {
  float s = 0.f;
  for (int k = 0; k < n; ++k) s += v[k];
  const float mean = warp_sum(s) / S;
  float q = 0.f;
  for (int k = 0; k < n; ++k) { const float dd = v[k] - mean; q += dd * dd; }
  const float rstd = rsqrtf(warp_sum(q) / S + eps);
  for (int k = 0; k < n; ++k) { const int h = lane + 32 * k; v[k] = (v[k] - mean) * rstd * g[h] + bt[h]; }
}

// One warp per token (graph_kernels.cu's entity_kernel at the seats' width): injection
// (x = relu(LN(u + gb)) * present) or a residual step (x = (x + relu(u + u2 + ub)) * present),
// then z = LN(x) when z is set.
__global__ void step_kernel(Step a) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (w >= a.count) return;
  const int n = (a.S - lane + 31) / 32;  // this lane's units: lane, lane + 32, ...
  float v[SPL];
  const float m = a.present[w];
  if (a.gb) {
    const float* gb = a.gb + (long)(w / a.P) * a.gbs;
    for (int k = 0; k < n; ++k) { const int h = lane + 32 * k; v[k] = a.u[(long)w * a.us + h] + gb[h]; }
    seat_norm(v, n, a.S, a.inj_g, a.inj_b, a.inj_eps, lane);
    for (int k = 0; k < n; ++k) v[k] = fmaxf(v[k], 0.f) * m;
  } else {
    for (int k = 0; k < n; ++k) {
      const int h = lane + 32 * k;
      v[k] = a.x[(long)w * a.S + h];
      if (a.u) v[k] = (v[k] + fmaxf(a.u[(long)w * a.us + h] + a.u2[(long)w * a.S + h] + a.ub[h], 0.f)) * m;
    }
  }
  if (a.gb || a.u)
    for (int k = 0; k < n; ++k) a.x[(long)w * a.S + lane + 32 * k] = v[k];
  if (a.z) {
    seat_norm(v, n, a.S, a.g, a.bt, a.eps, lane);
    for (int k = 0; k < n; ++k) a.z[(long)w * a.S + lane + 32 * k] = v[k];
  }
}

template <typename F>
struct Scatter {
  const float* msg; int ms;  // [B*P, ms] the seats' messages: NR * D to the nodes, D to the edges, D to the tiles
  const float* wN; const float* wE; const float* wT;
  F* xn; int sn;             // each type's GEMM input, at its first message column (row stride s*)
  F* xe; int se;
  F* xt; int st;
  int N, T, E, P, D;
  int* flag;
};

// One block per row: each node's, edge's and tile's sum of its seats' messages, by relation
// (the weights are relations_kernel's), into the columns of its GEMM input after the board's.
template <typename F>
__global__ void scatter_kernel(Scatter<F> a) {
  const int b = blockIdx.x;
  const float* msg = a.msg + (long)b * a.P * a.ms;
  const int nodes = a.N * NR * a.D, edges = a.E * a.D, tiles = a.T * a.D;
  bool bad = false;
  for (int j = threadIdx.x; j < nodes + edges + tiles; j += blockDim.x) {
    const float* w;
    F* out;
    int c;  // the message's column
    if (j < nodes) {
      const long i = (long)b * a.N + j / (NR * a.D);
      c = j % (NR * a.D);
      w = a.wN + i * NR * a.P + (c / a.D) * a.P;
      out = a.xn + i * a.sn + c;
    } else if (j < nodes + edges) {
      const int k = j - nodes;
      const long i = (long)b * a.E + k / a.D;
      c = NR * a.D + k % a.D;
      w = a.wE + i * a.P;
      out = a.xe + i * a.se + k % a.D;
    } else {
      const int k = j - nodes - edges;
      const long i = (long)b * a.T + k / a.D;
      c = (NR + 1) * a.D + k % a.D;
      w = a.wT + i * a.P;
      out = a.xt + i * a.st + k % a.D;
    }
    float acc = 0.f;
    for (int p = 0; p < a.P; ++p) acc += w[p] * msg[p * a.ms + c];
    store(out, acc, bad);
  }
  if (bad) atomicOr(a.flag, NONFINITE);
}

template <typename F>
struct Collect {
  const F* zn; int sn;       // the layer's LayerNorm'd board embeddings: the first H columns of
  const F* ze; int se;       // each type's GEMM input (row stride s*)
  const F* zt; int st;
  const float* wN; const float* wE; const float* wT;
  const float* den;          // [B, GROUPS * P]
  const float* wm;           // [GROUPS * D, H] node_messages | edge_messages | tile_messages
  const float* present;      // [B * P]
  const float* q; int qs;    // attention: [B*P, qs] queries | keys | values (A each); else the LN'd tokens [B*P, A]
  const float* bias;         // attention: [MAXP + 1, P, P, heads] by the seats present, from, to
  float scale; int heads;    // heads 0: the mean of the other seats' tokens (attention: false)
  int A;
  float* fb; int fs;         // [B*P, GROUPS * D + A]: the board's messages, then the seats'
  int N, T, E, P, D;
};

// One block per token, eight warps. Warps 0..6: the weighted mean of the token's nodes (one
// per relation), edges and tiles through that relation's message weights (linear: applied
// after the mean). Warp 7: what the token receives from the other seats' (multi-head attention
// over the present seats with the turn-order bias, or their mean), before its output weights.
template <typename F>
__global__ void collect_kernel(Collect<F> a) {
  const int token = blockIdx.x;
  const int lane = threadIdx.x & 31, group = threadIdx.x >> 5;
  const int b = token / a.P, p = token % a.P;
  float* fb = a.fb + (long)token * a.fs;
  const float* here = a.present + (long)b * a.P;
  if (group < GROUPS) {
    float acc[HPL] = {0.f, 0.f, 0.f};
    if (here[p] > 0.f) {  // an absent seat has no weights
      const int n = group < NR ? a.N : group == NR ? a.E : a.T;
      const F* z = group < NR ? a.zn : group == NR ? a.ze : a.zt;
      const int stride = group < NR ? a.sn : group == NR ? a.se : a.st;
      const float* w = group < NR ? a.wN + group * a.P + p : group == NR ? a.wE + p : a.wT + p;
      const int ws = group < NR ? NR * a.P : a.P;
      for (int i = 0; i < n; ++i) {
        const long e = (long)b * n + i;
        const float weight = w[e * ws];
        if (weight != 0.f)
          for (int k = 0; k < HPL; ++k) acc[k] += weight * load(z + e * stride + lane + 32 * k);
      }
    }
    const float scale = 1.f / a.den[(long)b * GROUPS * a.P + group * a.P + p];
    for (int d = 0; d < a.D; ++d) {
      const float* wm = a.wm + (group * a.D + d) * H;
      float dot = 0.f;
      for (int k = 0; k < HPL; ++k) dot += acc[k] * wm[lane + 32 * k];
      dot = warp_sum(dot);
      if (lane == 0) fb[group * a.D + d] = dot * scale;
    }
    return;
  }
  fb += GROUPS * a.D;
  float total;
  const int count = seats_present(here, a.P, total);
  for (int l = lane; l < a.A; l += 32) {
    float out = 0.f;
    if (here[p] <= 0.f) {  // an absent seat: nothing reads it, kept finite
    } else if (a.heads == 0) {
      float sum = 0.f;
      for (int j = 0; j < a.P; ++j) sum += a.q[((long)b * a.P + j) * a.qs + l] * here[j];
      out = (sum - a.q[(long)token * a.qs + l] * here[p]) / fmaxf(total - here[p], 1.f);
    } else {
      const int width = a.A / a.heads, head = l / width;
      const float* query = a.q + (long)token * a.qs + head * width;
      const float* bias = a.bias + (((long)count * a.P + p) * a.P) * a.heads + head;
      float score[MAXP];
      float most = -INFINITY;
      for (int j = 0; j < a.P; ++j) {
        if (here[j] <= 0.f) continue;
        const float* key = a.q + ((long)b * a.P + j) * a.qs + a.A + head * width;
        float dot = 0.f;
        for (int c = 0; c < width; ++c) dot += query[c] * key[c];
        score[j] = dot * a.scale + bias[j * a.heads];
        most = fmaxf(most, score[j]);
      }
      float norm = 0.f;
      for (int j = 0; j < a.P; ++j) {
        if (here[j] <= 0.f) continue;
        const float e = expf(score[j] - most);
        norm += e;
        out += e * a.q[((long)b * a.P + j) * a.qs + 2 * a.A + l];
      }
      out /= norm;
    }
    fb[l] = out;
  }
}

const float* ptr(const c10::optional<torch::Tensor>& t) { return t ? t->data_ptr<float>() : nullptr; }

}  // namespace

// l: N, T, E, P, S, the raw blocks' offsets (global, players, nodes, tiles, edges), the raw
// widths (players, nodes, tiles, edges), the valid features (nodes, tiles, edges) and present,
// then the table's counts kn, kt, ktd (-1 without the dice input), ke, kp, kg, kd, ks. The
// table holds raw columns in this order: node_plain[kn], settlement[P], city[P], distance[P],
// tile_plain[kt], buildings[P], (tile_odds[P], tile_dice[ktd],) edge_plain[ke], road[P],
// player[kp], global_plain[kg], global_dice[kd], seat_dice[P * ks]. bands: seat.DISTANCE_*.
// Returns xn, xt, xe, xs, gin, present, wN, wE, wT, den.
std::vector<torch::Tensor> relations(torch::Tensor obs, std::vector<int64_t> l, std::vector<double> bands,
                                     torch::Tensor tab) {
  Seats d{};
  d.B = (int)obs.size(0); d.R = (int)obs.size(1);
  d.N = (int)l[0]; d.T = (int)l[1]; d.E = (int)l[2]; d.P = (int)l[3]; d.S = (int)l[4];
  d.off_global = (int)l[5]; d.off_players = (int)l[6]; d.off_nodes = (int)l[7]; d.off_tiles = (int)l[8];
  d.off_edges = (int)l[9];
  d.wp = (int)l[10]; d.wn = (int)l[11]; d.wt = (int)l[12]; d.we = (int)l[13];
  d.vn = (int)l[14]; d.vt = (int)l[15]; d.ve = (int)l[16]; d.present = (int)l[17];
  d.kn = (int)l[18]; d.kt = (int)l[19]; d.ktd = (int)l[20]; d.ke = (int)l[21]; d.kp = (int)l[22];
  d.kg = (int)l[23]; d.kd = (int)l[24]; d.ks = (int)l[25];
  TORCH_CHECK(d.P >= 1 && d.P <= MAXP, "the seat kernels read up to ", MAXP, " seat slots");
  const bool dice = d.ktd >= 0;
  int at = 0;
  auto take = [&at](int n) { const int start = at; at += n; return start; };
  d.node_plain = take(d.kn); d.settlement = take(d.P); d.city = take(d.P); d.distance = take(d.P);
  d.tile_plain = take(d.kt); d.buildings = take(d.P);
  d.tile_odds = take(dice ? d.P : 0); d.tile_dice = take(dice ? d.ktd : 0);
  d.edge_plain = take(d.ke); d.road = take(d.P);
  d.player = take(d.kp);
  d.global_plain = take(d.kg); d.global_dice = take(d.kd); d.seat_dice = take(d.P * d.ks);
  TORCH_CHECK(tab.numel() == at && tab.scalar_type() == torch::kInt32, "the seat kernels' column table: ", at,
              " int32 columns expected");
  d.on = (float)bands[0]; d.one = (float)bands[1]; d.near = (float)bands[2];
  auto f = obs.options();
  auto xn = torch::empty({d.B * d.N, d.kn + 6}, f);
  auto xt = torch::empty({d.B * d.T, d.kt + 2 + (dice ? 2 + d.ktd : 0)}, f);
  auto xe = torch::empty({d.B * d.E, d.ke + 2}, f);
  auto xs = torch::empty({d.B * d.P, d.kp + d.ks}, f);
  auto gin = torch::empty({d.B, d.kg + 3 * d.S + d.kd}, f);
  auto present = torch::empty({d.B * d.P}, f);
  auto wN = torch::empty({d.B * d.N, NR * d.P}, f), wE = torch::empty({d.B * d.E, d.P}, f);
  auto wT = torch::empty({d.B * d.T, d.P}, f), den = torch::empty({d.B, GROUPS * d.P}, f);
  relations_kernel<<<d.B, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      obs.data_ptr<float>(), tab.data_ptr<int>(), d, xn.data_ptr<float>(), xt.data_ptr<float>(),
      xe.data_ptr<float>(), xs.data_ptr<float>(), gin.data_ptr<float>(), present.data_ptr<float>(),
      wN.data_ptr<float>(), wE.data_ptr<float>(), wT.data_ptr<float>(), den.data_ptr<float>());
  return {xn, xt, xe, xs, gin, present, wN, wE, wT, den};
}

// ps: [B*P, S] the seat projection of the seats' features; turn: [MAXP + 1, P, S] the turn-order
// features' term by the seats present and the slot (bias in). Writes the tokens' summary into
// columns offset.. of gin and returns the tokens [B*P, S].
torch::Tensor tokens(torch::Tensor ps, torch::Tensor turn, torch::Tensor present, torch::Tensor gin, int64_t P,
                     int64_t offset) {
  const int B = gin.size(0), S = ps.size(1);
  TORCH_CHECK(turn.numel() == (MAXP + 1) * P * S, "the turn-order table is [", MAXP + 1, ", P, S]");
  auto seats = torch::empty_like(ps);
  tokens_kernel<<<B, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
      ps.data_ptr<float>(), turn.data_ptr<float>(), present.data_ptr<float>(), (int)P, S, seats.data_ptr<float>(),
      gin.data_ptr<float>(), (int)gin.size(1), (int)offset);
  return seats;
}

// The tokens' injection (u, gb, inj_*) or residual step (u's first S columns + u2 + ub, all or
// none), then z = LN(x) when z is set; as graph_kernels.cu's entities for one type.
void step(torch::Tensor x, c10::optional<torch::Tensor> u, c10::optional<torch::Tensor> u2,
          c10::optional<torch::Tensor> ub, c10::optional<torch::Tensor> gb, int64_t gbs,
          c10::optional<torch::Tensor> inj_g, c10::optional<torch::Tensor> inj_b, double inj_eps,
          torch::Tensor present, c10::optional<torch::Tensor> z, c10::optional<torch::Tensor> g,
          c10::optional<torch::Tensor> bt, double eps, int64_t P) {
  const int S = x.size(1);
  TORCH_CHECK(S <= 32 * SPL, "the seat kernels are built for seat_hidden up to ", 32 * SPL);
  Step a{x.data_ptr<float>(), ptr(u), u ? (int)u->size(1) : 0, ptr(u2), ptr(ub), ptr(gb), (int)gbs, ptr(inj_g),
         ptr(inj_b), (float)inj_eps, present.data_ptr<float>(), z ? z->data_ptr<float>() : nullptr, ptr(g), ptr(bt),
         (float)eps,
         (int)present.numel(), S, (int)P};
  const int threads = 256;
  step_kernel<<<(a.count * 32 + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(a);
}

// msg: [B*P, >= (NR + 2) * D] the seats' messages (a view at their first column). xn, xe, xt:
// each type's GEMM input (float or half, all three alike), written from column on / oe / ot.
// flag: the kernels' flag (set when a half written is not finite), with half inputs.
void scatter(torch::Tensor msg, torch::Tensor wN, torch::Tensor wE, torch::Tensor wT, torch::Tensor xn, int64_t on,
             torch::Tensor xe, int64_t oe, torch::Tensor xt, int64_t ot, int64_t P, int64_t D,
             c10::optional<torch::Tensor> flag) {
  const int B = msg.size(0) / P;
  const int N = wN.size(0) / B, E = wE.size(0) / B, T = wT.size(0) / B;
  const int ms = msg.stride(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (xn.scalar_type() == torch::kHalf) {
    TORCH_CHECK(flag.has_value(), "half inputs need the flag");
    Scatter<__half> a{msg.data_ptr<float>(), ms, wN.data_ptr<float>(), wE.data_ptr<float>(), wT.data_ptr<float>(),
                      (__half*)xn.data_ptr() + on, (int)xn.size(1), (__half*)xe.data_ptr() + oe, (int)xe.size(1),
                      (__half*)xt.data_ptr() + ot, (int)xt.size(1), N, T, E, (int)P, (int)D, flag->data_ptr<int>()};
    scatter_kernel<__half><<<B, 256, 0, stream>>>(a);
  } else {
    Scatter<float> a{msg.data_ptr<float>(), ms, wN.data_ptr<float>(), wE.data_ptr<float>(), wT.data_ptr<float>(),
                     xn.data_ptr<float>() + on, (int)xn.size(1), xe.data_ptr<float>() + oe, (int)xe.size(1),
                     xt.data_ptr<float>() + ot, (int)xt.size(1), N, T, E, (int)P, (int)D, nullptr};
    scatter_kernel<float><<<B, 256, 0, stream>>>(a);
  }
}

// zn, ze, zt: each type's GEMM input (float or half), whose first H columns are the layer's
// LayerNorm'd embeddings. q: [B*P, >= 3 * A] queries | keys | values (a view at the queries'
// first column) with heads >= 1, or the LN'd tokens [B*P, A] with heads 0. Writes fb
// [B*P, (NR + 2) * D + A].
void collect(torch::Tensor zn, torch::Tensor ze, torch::Tensor zt, torch::Tensor wN, torch::Tensor wE,
             torch::Tensor wT, torch::Tensor den, torch::Tensor wm, torch::Tensor present, torch::Tensor q,
             c10::optional<torch::Tensor> bias, double scale, int64_t heads, int64_t A, torch::Tensor fb,
             int64_t P) {
  const int tokens = present.numel(), B = tokens / P;
  const int N = wN.size(0) / B, E = wE.size(0) / B, T = wT.size(0) / B, D = wm.size(0) / GROUPS;
  TORCH_CHECK(wm.size(1) == H && fb.size(1) == GROUPS * D + A, "the seat kernels' message widths");
  TORCH_CHECK(heads == 0 || (bias.has_value() && A % heads == 0), "attention needs its bias and whole heads");
  auto stream = at::cuda::getCurrentCUDAStream();
  if (zn.scalar_type() == torch::kHalf) {
    Collect<__half> a{(const __half*)zn.data_ptr(), (int)zn.size(1), (const __half*)ze.data_ptr(), (int)ze.size(1),
                      (const __half*)zt.data_ptr(), (int)zt.size(1), wN.data_ptr<float>(), wE.data_ptr<float>(),
                      wT.data_ptr<float>(), den.data_ptr<float>(), wm.data_ptr<float>(), present.data_ptr<float>(),
                      q.data_ptr<float>(), (int)q.stride(0), ptr(bias), (float)scale, (int)heads, (int)A,
                      fb.data_ptr<float>(), (int)fb.size(1), N, T, E, (int)P, D};
    collect_kernel<__half><<<tokens, 256, 0, stream>>>(a);
  } else {
    Collect<float> a{zn.data_ptr<float>(), (int)zn.size(1), ze.data_ptr<float>(), (int)ze.size(1),
                     zt.data_ptr<float>(), (int)zt.size(1), wN.data_ptr<float>(), wE.data_ptr<float>(),
                     wT.data_ptr<float>(), den.data_ptr<float>(), wm.data_ptr<float>(), present.data_ptr<float>(),
                     q.data_ptr<float>(), (int)q.stride(0), ptr(bias), (float)scale, (int)heads, (int)A,
                     fb.data_ptr<float>(), (int)fb.size(1), N, T, E, (int)P, D};
    collect_kernel<float><<<tokens, 256, 0, stream>>>(a);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relations", &relations);
  m.def("tokens", &tokens);
  m.def("step", &step);
  m.def("scatter", &scatter);
  m.def("collect", &collect);
  m.attr("hidden") = H;
  m.attr("max_seat_hidden") = 32 * SPL;
}
