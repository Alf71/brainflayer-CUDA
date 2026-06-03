#include "Kernel.cuh"
#include "filter.h"
#include "lib/hash/sha256.h"

#include <algorithm>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

extern thread_local int DEVICE_NR;
extern bool g_save_output_with_gpu_prefix;
extern std::atomic<uint64_t> g_runtime_founds;
extern void blake2b(const uint8_t* input, size_t inlen, uint8_t* out, size_t outlen);

static std::mutex g_output_mutex;
static std::mutex g_found_mutex;
static bool Silent = false;
static std::atomic<int> g_cpu_postcheck_suppression{ 0 };

bool STOP_THREAD = false;
std::atomic<uint64_t> false_positive{ 0 };

struct AsyncSaveTask {
    std::function<void()> fn;
    const char* family = nullptr;
    size_t batch_size = 0;
    bool save = false;
    int gpu_id = -1;
};

struct AsyncSaveQueue {
    std::mutex mutex;
    std::condition_variable cv_not_empty;
    std::condition_variable cv_not_full;
    std::condition_variable cv_idle;
    std::deque<AsyncSaveTask> tasks;
    std::thread worker;
    size_t in_flight = 0;
    bool stopping = false;
    int gpu_id = -1;
};

static std::mutex g_async_save_queues_mutex;
static std::unordered_map<int, std::shared_ptr<AsyncSaveQueue>> g_async_save_queues;
static constexpr size_t MAX_PENDING_SAVE_TASKS_PER_GPU = 64u;

static thread_local int g_output_gpu = -1;
static thread_local bool g_stdout_line_start = true;
static thread_local bool g_file_line_start = true;

__host__ void setSilentMode()
{
    Silent = true;
}

__host__ void push_save_cpu_postcheck_suppression()
{
    g_cpu_postcheck_suppression.fetch_add(1, std::memory_order_acq_rel);
}

__host__ void pop_save_cpu_postcheck_suppression()
{
    const int prev = g_cpu_postcheck_suppression.fetch_sub(1, std::memory_order_acq_rel);
    if (prev <= 0) {
        g_cpu_postcheck_suppression.store(0, std::memory_order_release);
    }
}

static void set_output_gpu_prefix(int gpu_id)
{
    g_output_gpu = gpu_id;
    g_stdout_line_start = true;
    g_file_line_start = true;
}

static std::string apply_gpu_prefix(FILE* stream, const std::string& text)
{
    if (g_output_gpu < 0 || stream == nullptr || stream == stderr || text.empty()) {
        return text;
    }

    bool* line_start = (stream == stdout) ? &g_stdout_line_start : &g_file_line_start;
    const std::string prefix = "GPU " + std::to_string(g_output_gpu) + ":";
    std::string out;
    out.reserve(text.size() + prefix.size() * 2u);
    for (char ch : text) {
        if (*line_start && ch != '\n') {
            out.append(prefix);
            *line_start = false;
        }
        out.push_back(ch);
        if (ch == '\n') {
            *line_start = true;
        }
    }
    return out;
}

static int save_output_write_prebuilt(FILE* stream, const std::string& line)
{
    if (stream == nullptr || line.empty()) {
        return 0;
    }

    std::string normalized = line;
    if (!normalized.empty() && normalized.front() == '\n') {
        normalized.erase(0, 1);
    }
    if (!normalized.empty() && normalized.back() != '\n') {
        normalized.push_back('\n');
    }
    std::lock_guard<std::mutex> lock(g_output_mutex);
    std::string out = apply_gpu_prefix(stream, normalized);
    const size_t written = std::fwrite(out.data(), 1, out.size(), stream);
    std::fflush(stream);
    return static_cast<int>(written);
}

static void save_output_flush(FILE* stream)
{
    if (stream == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_output_mutex);
    std::fflush(stream);
}

static void async_save_worker_loop(const std::shared_ptr<AsyncSaveQueue>& queue)
{
    set_output_gpu_prefix(g_save_output_with_gpu_prefix ? queue->gpu_id : -1);
    for (;;) {
        AsyncSaveTask task;
        {
            std::unique_lock<std::mutex> lock(queue->mutex);
            queue->cv_not_empty.wait(lock, [&]() { return queue->stopping || !queue->tasks.empty(); });
            if (queue->stopping && queue->tasks.empty()) {
                break;
            }
            task = std::move(queue->tasks.front());
            queue->tasks.pop_front();
            ++queue->in_flight;
            queue->cv_not_full.notify_one();
        }

        task.fn();

        {
            std::lock_guard<std::mutex> lock(queue->mutex);
            if (queue->in_flight > 0) {
                --queue->in_flight;
            }
            if (queue->tasks.empty() && queue->in_flight == 0) {
                queue->cv_idle.notify_all();
            }
        }
    }
    set_output_gpu_prefix(-1);
}

static std::shared_ptr<AsyncSaveQueue> get_async_save_queue_for_gpu(int gpu_id)
{
    std::lock_guard<std::mutex> map_lock(g_async_save_queues_mutex);
    auto it = g_async_save_queues.find(gpu_id);
    if (it != g_async_save_queues.end()) {
        return it->second;
    }

    auto queue = std::make_shared<AsyncSaveQueue>();
    queue->gpu_id = gpu_id;
    queue->worker = std::thread([queue]() { async_save_worker_loop(queue); });
    g_async_save_queues.emplace(gpu_id, queue);
    return queue;
}

template <typename Fn, typename... Args>
static void enqueue_async_save_task_for_current_gpu(const char* family, size_t batch_size, bool save, Fn&& fn, Args&&... args)
{
    const int gpu_id = DEVICE_NR;
    auto queue = get_async_save_queue_for_gpu(gpu_id);
    AsyncSaveTask task;
    task.family = family;
    task.batch_size = batch_size;
    task.save = save;
    task.gpu_id = gpu_id;
    task.fn = std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...);

    std::unique_lock<std::mutex> lock(queue->mutex);
    queue->cv_not_full.wait(lock, [&]() {
        return queue->stopping || queue->tasks.size() < MAX_PENDING_SAVE_TASKS_PER_GPU;
    });
    if (queue->stopping) {
        throw std::runtime_error("async save queue is stopping");
    }
    queue->tasks.push_back(std::move(task));
    lock.unlock();
    queue->cv_not_empty.notify_one();
}

static void wait_for_async_save_queue(const std::shared_ptr<AsyncSaveQueue>& queue)
{
    if (!queue) {
        return;
    }
    std::unique_lock<std::mutex> lock(queue->mutex);
    queue->cv_idle.wait(lock, [&]() { return queue->tasks.empty() && queue->in_flight == 0; });
}

__host__ void wait_for_current_gpu_async_save_queue()
{
    std::shared_ptr<AsyncSaveQueue> queue;
    {
        std::lock_guard<std::mutex> map_lock(g_async_save_queues_mutex);
        auto it = g_async_save_queues.find(DEVICE_NR);
        if (it != g_async_save_queues.end()) {
            queue = it->second;
        }
    }
    wait_for_async_save_queue(queue);
}

__host__ void flush_all_async_save_queues()
{
    std::vector<std::shared_ptr<AsyncSaveQueue>> queues;
    {
        std::lock_guard<std::mutex> map_lock(g_async_save_queues_mutex);
        for (const auto& it : g_async_save_queues) {
            queues.push_back(it.second);
        }
    }
    for (const auto& queue : queues) {
        wait_for_async_save_queue(queue);
    }
}

__host__ void shutdown_async_save_queues()
{
    std::vector<std::shared_ptr<AsyncSaveQueue>> queues;
    {
        std::lock_guard<std::mutex> map_lock(g_async_save_queues_mutex);
        for (const auto& it : g_async_save_queues) {
            queues.push_back(it.second);
        }
    }
    for (const auto& queue : queues) {
        wait_for_async_save_queue(queue);
        {
            std::lock_guard<std::mutex> lock(queue->mutex);
            queue->stopping = true;
        }
        queue->cv_not_empty.notify_all();
        queue->cv_not_full.notify_all();
    }
    for (const auto& queue : queues) {
        if (queue && queue->worker.joinable()) {
            queue->worker.join();
        }
    }
    std::lock_guard<std::mutex> map_lock(g_async_save_queues_mutex);
    g_async_save_queues.clear();
}

static inline std::string hex_bytes(const uint8_t* bytes, size_t len)
{
    static const char* h = "0123456789abcdef";
    std::string out;
    out.resize(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out[i * 2] = h[bytes[i] >> 4];
        out[i * 2 + 1] = h[bytes[i] & 0x0f];
    }
    return out;
}

static inline uint32_t safe_found_string_len(const char* data, uint32_t reported_len, uint32_t cap = 511)
{
    if (reported_len > 0 && reported_len <= cap) {
        return reported_len;
    }
    uint32_t n = 0;
    while (n < cap && data[n] != '\0') {
        ++n;
    }
    return n;
}

static bool cpu_postcheck(const uint32_t hash_words[20])
{
    if (g_cpu_postcheck_suppression.load(std::memory_order_acquire) > 0) {
        return true;
    }
    if (!useBloomCPU && !useXorCPU) {
        return true;
    }
    hash160_t h{};
    std::memcpy(h.uc, hash_words, 32);
    return find_in_bloom(h);
}

static inline int cmp_be(const uint8_t* x, const uint8_t* y, size_t n) {
    for (size_t i = 0; i < n; ++i) { if (x[i] != y[i]) return (x[i] < y[i]) ? -1 : 1; }
    return 0;
}

static inline void sub_be(const uint8_t* x, const uint8_t* y, uint8_t* out, size_t n) {
    int b = 0; for (size_t i = n; i-- > 0;) { int t = (int)x[i] - (int)y[i] - b; if (t < 0) { t += 256; b = 1; } else b = 0; out[i] = (uint8_t)t; }
}

static inline void add_be(const uint8_t* x, const uint8_t* y, uint8_t* out, size_t n) {
    int c = 0; for (size_t i = n; i-- > 0;) { int t = (int)x[i] + (int)y[i] + c; out[i] = (uint8_t)(t & 0xFF); c = t >> 8; }
}

static const uint8_t SECP256K1_N[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};
static const uint8_t SECP256K1_LAMBDA[32] = {
    0x53,0x63,0xAD,0x4C,0xC0,0x5C,0x30,0xE0,0xA5,0x26,0x1C,0x02,0x88,0x12,0x64,0x5A,
    0x12,0x2E,0x22,0xEA,0x20,0x81,0x66,0x78,0xDF,0x02,0x96,0x7C,0x1B,0x23,0xBD,0x72
};
static const uint8_t SECP256K1_LAMBDA2[32] = {
    0xAC,0x9C,0x52,0xB3,0x3F,0xA3,0xAD,0x48,0x58,0xA6,0x05,0x9F,0x6D,0x6C,0x42,0x98,
    0xD4,0xE4,0x4B,0x54,0x91,0x04,0x33,0xB1,0xF8,0xC0,0x06,0x2D,0xD3,0x1E,0x8C,0xC0
};

static void scalar_mult_mod_n_256(const uint8_t k_be[32], const uint8_t mult_be[32], uint8_t out_be[32]) {
    uint8_t r[33] = {0}, t[33];
    uint8_t n_pad[33]; memcpy(n_pad, SECP256K1_N, 32); n_pad[32] = 0;
    uint8_t mult_pad[33]; memcpy(mult_pad, mult_be, 32); mult_pad[32] = 0;
    for (int bi = 0; bi < 256; bi++) {
        int by = bi / 8, br = 7 - (bi % 8);
        int set = (k_be[by] >> br) & 1;
        add_be(r, r, t, 33);
        memcpy(r, t, 33);
        if (set) {
            add_be(r, mult_pad, t, 33);
            memcpy(r, t, 33);
        }
        while (cmp_be(r, n_pad, 33) >= 0) {
            sub_be(r, n_pad, t, 33);
            memcpy(r, t, 33);
        }
    }
    memcpy(out_be, r, 32);
}

static inline bool is_zero_be32(const uint8_t* x) {
    for (int i = 0; i < 32; ++i) {
        if (x[i] != 0) return false;
    }
    return true;
}

static inline void negate_mod_n_256(const uint8_t in_be[32], uint8_t out_be[32]) {
    if (is_zero_be32(in_be)) {
        memset(out_be, 0, 32);
        return;
    }
    sub_be(SECP256K1_N, in_be, out_be, 32);
}

static inline bool decode_endo_tag(uint8_t type, uint8_t& base_type, uint8_t& lambda_sel, bool& negate) {
    if (type < ENDO_TAG_BASE) return false;
    const uint8_t code = (uint8_t)(type - ENDO_TAG_BASE);
    const uint8_t group = (uint8_t)(code / ENDO_GROUP_STRIDE);
    const uint8_t variant = (uint8_t)(code % ENDO_GROUP_STRIDE);
    if (group > ENDO_GROUP_XPOINT) return false;
    if (variant > ENDO_VARIANT_ENDO2_NEG) return false;

    switch (group) {
    case ENDO_GROUP_COMPRESSED: base_type = 0x02; break;
    case ENDO_GROUP_SEGWIT: base_type = 0x03; break;
    case ENDO_GROUP_UNCOMPRESSED: base_type = 0x01; break;
    case ENDO_GROUP_ETH: base_type = 0x06; break;
    case ENDO_GROUP_TAPROOT: base_type = 0x04; break;
    case ENDO_GROUP_XPOINT: base_type = 0x05; break;
    default: return false;
    }

    lambda_sel = 0;
    negate = false;
    switch (variant) {
    case ENDO_VARIANT_BASE_POS:
        break;
    case ENDO_VARIANT_BASE_NEG:
        negate = true;
        break;
    case ENDO_VARIANT_ENDO1_POS:
        lambda_sel = 1;
        break;
    case ENDO_VARIANT_ENDO1_NEG:
        lambda_sel = 1;
        negate = true;
        break;
    case ENDO_VARIANT_ENDO2_POS:
        lambda_sel = 2;
        break;
    case ENDO_VARIANT_ENDO2_NEG:
        lambda_sel = 2;
        negate = true;
        break;
    default:
        return false;
    }
    return true;
}

static inline uint8_t endo_base_type_or_self(uint8_t type) {
    uint8_t base_type = type;
    uint8_t lambda_sel = 0;
    bool negate = false;
    if (decode_endo_tag(type, base_type, lambda_sel, negate)) {
        return base_type;
    }
    return type;
}

static void fix_endo_key(const uint8_t key_in[32], uint8_t key_out[32], uint8_t endo_type) {
    uint8_t base_type = 0;
    uint8_t lambda_sel = 0;
    bool negate = false;
    if (!decode_endo_tag(endo_type, base_type, lambda_sel, negate)) {
        if (endo_type == 0xC2) {
            scalar_mult_mod_n_256(key_in, SECP256K1_LAMBDA2, key_out);
            return;
        }
        memcpy(key_out, key_in, 32);
        return;
    }

    uint8_t tmp[32];
    if (lambda_sel == 1) scalar_mult_mod_n_256(key_in, SECP256K1_LAMBDA, tmp);
    else if (lambda_sel == 2) scalar_mult_mod_n_256(key_in, SECP256K1_LAMBDA2, tmp);
    else memcpy(tmp, key_in, 32);

    if (negate) negate_mod_n_256(tmp, key_out);
    else memcpy(key_out, tmp, 32);
}

static inline const char* byte_to_coin(uint8_t coin, bool& ed)
{
    switch (coin) {
    case 0x01: return "UNCOMPRESSED";
    case 0x02: return "COMPRESSED";
    case 0x03: return "SEGWIT";
    case 0x04: return "TAPROOT";
    case 0x05: return "XPOINT";
    case 0x06: return "ETH";
    case 0x20: ed = true; return "APTOS(ed25519)";
    case 0x21: ed = true; return "APTOS(Generalized ed25519)";
    case 0x22: return "APTOS(Generalized secp256k1)";
    case 0x30: ed = true; return "DOT(ed25519)";
    case 0x31: ed = true; return "DOT(sr25519)";
    case 0x41: return "FIL(f1)";
    case 0x42: return "FIL(f4)";
    case 0x50: return "IOTA(secp256k1)";
    case 0x51: ed = true; return "IOTA(ed25519)";
    case 0x52: ed = true; return "ICP(ed25519)";
    case 0x53: return "ICP(secp256k1)";
    case 0x60: ed = true; return "SOLANA";
    case 0x70: return "SUI(secp256k1)";
    case 0x71: ed = true; return "SUI(ed25519)";
    case 0x80: ed = true; return "TON(v1r1)";
    case 0x81: ed = true; return "TON(v1r2)";
    case 0x82: ed = true; return "TON(v1r3)";
    case 0x83: ed = true; return "TON(v2r1)";
    case 0x84: ed = true; return "TON(v2r2)";
    case 0x85: ed = true; return "TON(v3r1)";
    case 0x86: ed = true; return "TON(v3r2)";
    case 0x87: ed = true; return "TON(v4r1)";
    case 0x88: ed = true; return "TON(v4r2)";
    case 0x89: ed = true; return "TON(v5r1)";
    case 0x8a: ed = true; return "TON(hv1)";
    case 0x8b: ed = true; return "TON(hv2)";
    case 0x8c: ed = true; return "TON(hv3)";
    case 0x90: return "XRP(secp256k1)";
    case 0x91: ed = true; return "XRP(ed25519)";
    case 0x92: return "XTZ(secp256k1)";
    case 0x93: ed = true; return "XTZ(ed25519)";
    default: return "UNKNOWN";
    }
}

const char* const ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const char* const RIPPLE_ALPHABET = "rpshnaf39wBUDNEGHJKLM4PQRST7VWXYZ2bcdeCg65jkm8oFqi1tuvAxyz";

std::string encodeBase58_d(const uint8_t* bytes, size_t length)
{
    if (bytes == nullptr || length == 0) {
        return std::string();
    }
    size_t zeros = 0;
    while (zeros < length && bytes[zeros] == 0) {
        ++zeros;
    }
    std::vector<uint8_t> b58;
    b58.reserve((length - zeros) * 138 / 100 + 1);
    for (size_t i = zeros; i < length; ++i) {
        uint32_t carry = bytes[i];
        for (size_t j = 0; j < b58.size(); ++j) {
            uint32_t x = static_cast<uint32_t>(b58[j]) * 256u + carry;
            b58[j] = static_cast<uint8_t>(x % 58u);
            carry = x / 58u;
        }
        while (carry > 0) {
            b58.push_back(static_cast<uint8_t>(carry % 58u));
            carry /= 58u;
        }
    }
    std::string out(zeros, ALPHABET[0]);
    for (size_t i = 0; i < b58.size(); ++i) {
        out.push_back(ALPHABET[b58[b58.size() - 1 - i]]);
    }
    return out;
}

std::string encodeBase58_d_XRP(const uint8_t* bytes, size_t length)
{
    if (bytes == nullptr || length == 0) {
        return std::string();
    }
    size_t zeros = 0;
    while (zeros < length && bytes[zeros] == 0) {
        ++zeros;
    }
    std::vector<uint8_t> b58;
    b58.reserve((length - zeros) * 138 / 100 + 1);
    for (size_t i = zeros; i < length; ++i) {
        uint32_t carry = bytes[i];
        for (size_t j = 0; j < b58.size(); ++j) {
            uint32_t x = static_cast<uint32_t>(b58[j]) * 256u + carry;
            b58[j] = static_cast<uint8_t>(x % 58u);
            carry = x / 58u;
        }
        while (carry > 0) {
            b58.push_back(static_cast<uint8_t>(carry % 58u));
            carry /= 58u;
        }
    }
    std::string out(zeros, RIPPLE_ALPHABET[0]);
    for (size_t i = 0; i < b58.size(); ++i) {
        out.push_back(RIPPLE_ALPHABET[b58[b58.size() - 1 - i]]);
    }
    return out;
}

std::string hash160ToBase58_d(const uint8_t hash160[20], uint8_t prefix)
{
    uint8_t extended[25]{};
    extended[0] = prefix;
    std::memcpy(extended + 1, hash160, 20);
    uint8_t hash[32];
    sha256(extended, 21, hash);
    sha256(hash, 32, hash);
    std::memcpy(extended + 21, hash, 4);
    return encodeBase58_d(extended, 25);
}

std::string hash160ToXRP(const uint8_t hash160[20], uint8_t prefix)
{
    uint8_t extended[25]{};
    extended[0] = prefix;
    std::memcpy(extended + 1, hash160, 20);
    uint8_t hash[32];
    sha256(extended, 21, hash);
    sha256(hash, 32, hash);
    std::memcpy(extended + 21, hash, 4);
    return encodeBase58_d_XRP(extended, 25);
}

uint32_t bech32_polymod_step_d(uint32_t pre)
{
    const uint8_t b = pre >> 25;
    return ((pre & 0x1ffffffu) << 5) ^
        (-((b >> 0) & 1) & 0x3b6a57b2u) ^
        (-((b >> 1) & 1) & 0x26508e6du) ^
        (-((b >> 2) & 1) & 0x1ea119fau) ^
        (-((b >> 3) & 1) & 0x3d4233ddu) ^
        (-((b >> 4) & 1) & 0x2a1462b3u);
}

static int convert_bits(uint8_t* out, size_t* outlen, int outbits, const uint8_t* in, size_t inlen, int inbits, int pad)
{
    uint32_t val = 0;
    int bits = 0;
    const uint32_t maxv = (1u << outbits) - 1u;
    while (inlen--) {
        val = (val << inbits) | *(in++);
        bits += inbits;
        while (bits >= outbits) {
            bits -= outbits;
            out[(*outlen)++] = static_cast<uint8_t>((val >> bits) & maxv);
        }
    }
    if (pad && bits) {
        out[(*outlen)++] = static_cast<uint8_t>((val << (outbits - bits)) & maxv);
    }
    else if (!pad && (((val << (outbits - bits)) & maxv) || bits >= inbits)) {
        return 0;
    }
    return 1;
}

int bech32_encode(char* output, const char* hrp, const uint8_t* data, size_t data_len, bool bech32m = false)
{
    static const char* charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    uint32_t chk = 1;
    for (const char* p = hrp; *p; ++p) {
        const int ch = *p;
        if (ch < 33 || ch > 126 || (ch >= 'A' && ch <= 'Z')) {
            return 0;
        }
        chk = bech32_polymod_step_d(chk) ^ (ch >> 5);
    }
    chk = bech32_polymod_step_d(chk);
    while (*hrp != 0) {
        chk = bech32_polymod_step_d(chk) ^ (*hrp & 0x1f);
        *(output++) = *(hrp++);
    }
    *(output++) = '1';
    for (size_t i = 0; i < data_len; ++i) {
        if (data[i] >> 5) {
            return 0;
        }
        chk = bech32_polymod_step_d(chk) ^ data[i];
        *(output++) = charset[data[i]];
    }
    for (int i = 0; i < 6; ++i) {
        chk = bech32_polymod_step_d(chk);
    }
    chk ^= bech32m ? 0x2bc830a3u : 1u;
    for (int i = 0; i < 6; ++i) {
        *(output++) = charset[(chk >> ((5 - i) * 5)) & 0x1f];
    }
    *output = 0;
    return 1;
}

int segwit_addr_encode(char* output, const char* hrp, uint16_t witver, const uint8_t* witprog, size_t witprog_len)
{
    uint8_t data[65]{};
    size_t datalen = 0;
    if (witver > 16) {
        return 0;
    }
    if (witver == 0 && witprog_len != 20 && witprog_len != 32) {
        return 0;
    }
    if (witprog_len < 2 || witprog_len > 40) {
        return 0;
    }
    data[0] = static_cast<uint8_t>(witver);
    convert_bits(data + 1, &datalen, 5, witprog, witprog_len, 8, 1);
    ++datalen;
    return bech32_encode(output, hrp, data, datalen, witver != 0);
}

static uint16_t crc16(const std::vector<uint8_t>& data)
{
    uint16_t crc = 0;
    for (uint8_t byte : data) {
        crc ^= static_cast<uint16_t>(byte) << 8;
        for (int i = 0; i < 8; ++i) {
            crc = (crc & 0x8000u) ? static_cast<uint16_t>((crc << 1) ^ 0x1021u) : static_cast<uint16_t>(crc << 1);
        }
    }
    return crc;
}

std::string base64_encode(const std::vector<uint8_t>& input)
{
    static const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    out.reserve(((input.size() + 2u) / 3u) * 4u);
    size_t i = 0;
    while (i + 3u <= input.size()) {
        uint32_t v = (static_cast<uint32_t>(input[i]) << 16) |
            (static_cast<uint32_t>(input[i + 1]) << 8) |
            static_cast<uint32_t>(input[i + 2]);
        out.push_back(chars[(v >> 18) & 0x3f]);
        out.push_back(chars[(v >> 12) & 0x3f]);
        out.push_back(chars[(v >> 6) & 0x3f]);
        out.push_back(chars[v & 0x3f]);
        i += 3u;
    }
    if (i < input.size()) {
        uint32_t v = static_cast<uint32_t>(input[i]) << 16;
        const bool two = (i + 1u) < input.size();
        if (two) {
            v |= static_cast<uint32_t>(input[i + 1]) << 8;
        }
        out.push_back(chars[(v >> 18) & 0x3f]);
        out.push_back(chars[(v >> 12) & 0x3f]);
        out.push_back(two ? chars[(v >> 6) & 0x3f] : '=');
        out.push_back('=');
    }
    return out;
}

std::string generate_ton_address(const uint8_t* hash, bool is_testnet = false, bool is_bounceable = false, int8_t workchain = 0)
{
    std::vector<uint8_t> data;
    data.reserve(36);
    data.push_back(is_bounceable ? 0x11 : 0x51);
    if (is_testnet) {
        data[0] |= 0x80;
    }
    data.push_back(static_cast<uint8_t>(workchain));
    data.insert(data.end(), hash, hash + 32);
    const uint16_t crc = crc16(data);
    data.push_back(static_cast<uint8_t>(crc >> 8));
    data.push_back(static_cast<uint8_t>(crc & 0xffu));
    return base64_encode(data);
}

std::string polkadotAddress(const uint8_t pubKey[32])
{
    std::vector<uint8_t> data;
    data.reserve(35);
    data.push_back(0x00);
    data.insert(data.end(), pubKey, pubKey + 32);
    static const char prefix[] = "SS58PRE";
    std::vector<uint8_t> checksumInput;
    checksumInput.reserve((sizeof(prefix) - 1u) + data.size());
    checksumInput.insert(checksumInput.end(), prefix, prefix + sizeof(prefix) - 1u);
    checksumInput.insert(checksumInput.end(), data.begin(), data.end());
    uint8_t hash[64];
    blake2b(checksumInput.data(), checksumInput.size(), hash, sizeof(hash));
    data.push_back(hash[0]);
    data.push_back(hash[1]);
    return encodeBase58_d(data.data(), data.size());
}

int iota_bech32(char* out, const char* hrp, const uint8_t* raw_addr, uint8_t version)
{
    uint8_t data5[80];
    size_t data5_len = 0;
    uint8_t prepared[33];
    prepared[0] = version;
    std::memcpy(prepared + 1, raw_addr, 32);
    convert_bits(data5, &data5_len, 5, prepared, sizeof(prepared), 8, 1);
    return bech32_encode(out, hrp, data5, data5_len, false);
}

static size_t base32_lower_encode(const uint8_t* in, size_t inlen, char* out)
{
    static const char* alphabet = "abcdefghijklmnopqrstuvwxyz234567";
    size_t outlen = 0;
    uint32_t buffer = 0;
    int bits = 0;
    for (size_t i = 0; i < inlen; ++i) {
        buffer = (buffer << 8) | in[i];
        bits += 8;
        while (bits >= 5) {
            out[outlen++] = alphabet[(buffer >> (bits - 5)) & 0x1f];
            bits -= 5;
        }
    }
    if (bits > 0) {
        out[outlen++] = alphabet[(buffer << (5 - bits)) & 0x1f];
    }
    return outlen;
}

char* filecoin_from_payload(const uint8_t payload20[20], const char* network, char* out)
{
    uint8_t checksum[4];
    if (network[0] == 'f' && network[1] == '4') {
        uint8_t buf[22];
        buf[0] = 0x04;
        buf[1] = 0x0a;
        std::memcpy(buf + 2, payload20, 20);
        blake2b(buf, sizeof(buf), checksum, sizeof(checksum));
    }
    else {
        uint8_t buf[21];
        buf[0] = 0x01;
        std::memcpy(buf + 1, payload20, 20);
        blake2b(buf, sizeof(buf), checksum, sizeof(checksum));
    }
    uint8_t data[24];
    std::memcpy(data, payload20, 20);
    std::memcpy(data + 20, checksum, 4);
    char b32[40];
    const size_t n = base32_lower_encode(data, sizeof(data), b32);
    b32[n] = '\0';
    if (network[0] == 'f' && network[1] == '4') {
        out[0] = 'f'; out[1] = '4'; out[2] = '1'; out[3] = '0'; out[4] = 'f';
        std::memcpy(out + 5, b32, 39);
        out[44] = '\0';
        return out;
    }
    out[0] = network[0];
    out[1] = network[1];
    std::memcpy(out + 2, b32, 39);
    out[41] = '\0';
    return out;
}

static const uint8_t PREFIX_TZ1[3] = { 0x06, 0xA1, 0x9F };
static const uint8_t PREFIX_TZ2[3] = { 0x06, 0xA1, 0xA1 };

std::string tz_from_pkhash(const uint8_t prefix[3], const uint8_t pk_hash[20])
{
    uint8_t payload[23];
    std::memcpy(payload, prefix, 3);
    std::memcpy(payload + 3, pk_hash, 20);
    uint8_t d1[32], d2[32];
    sha256(payload, sizeof(payload), d1);
    sha256(d1, 32, d2);
    uint8_t finalbuf[27];
    std::memcpy(finalbuf, payload, sizeof(payload));
    std::memcpy(finalbuf + sizeof(payload), d2, 4);
    return encodeBase58_d(finalbuf, sizeof(finalbuf));
}

static uint32_t reverseByteOrderHost(uint32_t val)
{
    return ((val >> 24) & 0x000000ffu) |
        ((val >> 8) & 0x0000ff00u) |
        ((val << 8) & 0x00ff0000u) |
        ((val << 24) & 0xff000000u);
}

static void append_u32_hex_lower(std::string& out, uint32_t value)
{
    static const char kHex[] = "0123456789abcdef";
    for (int shift = 28; shift >= 0; shift -= 4) {
        out.push_back(kHex[(value >> shift) & 0x0f]);
    }
}

static std::string build_hash_payload_words(const uint32_t* words, size_t word_count, bool add_0x_prefix = false)
{
    std::string out;
    out.reserve((add_0x_prefix ? 2u : 0u) + word_count * 8u);
    if (add_0x_prefix) {
        out += "0x";
    }
    for (size_t idx = 0; idx < word_count; ++idx) {
        append_u32_hex_lower(out, reverseByteOrderHost(words[idx]));
    }
    return out;
}

static bool found_hash_is_32_bytes(uint8_t coin_type)
{
    switch (coin_type) {
    case 0x01:
    case 0x02:
    case 0x03:
    case 0x06:
    case 0x41:
    case 0x42:
    case 0x90:
    case 0x91:
    case 0x92:
    case 0x93:
        return false;
    default:
        return true;
    }
}

static std::string build_round_suffix(int64_t found_round)
{
    return (found_round == 0) ? std::string() : ((found_round > 0 ? " +" : " -") + std::to_string(found_round > 0 ? found_round : -found_round));
}

static std::string build_key_segment(const uint8_t* priv_key, int64_t round)
{
    std::string out = hex_bytes(priv_key, 32);
    out += build_round_suffix(round);
    return out;
}

static std::string build_result_line_with_newline(const std::string& head, const std::string& key_segment, const char* type, const std::string& payload, bool add_newline)
{
    std::string out;
    const size_t type_len = (type == nullptr) ? 0u : std::strlen(type);
    out.reserve(head.size() + key_segment.size() + type_len + payload.size() + (add_newline ? 3u : 2u));
    out += head;
    out += key_segment;
    out.push_back(':');
    if (type_len != 0u) {
        out.append(type, type_len);
    }
    out.push_back(':');
    out += payload;
    if (add_newline) {
        out.push_back('\n');
    }
    return out;
}

static void emit_mode_result_line(FILE* file,
    const std::string& file_head,
    const std::string& key_segment,
    const char* type,
    const std::string& payload,
    const std::string& stdout_head,
    bool silent,
    bool stdout_add_newline = true)
{
    (void)save_output_write_prebuilt(file, build_result_line_with_newline(file_head, key_segment, type, payload, true));
    if (!silent) {
        (void)save_output_write_prebuilt(stdout, build_result_line_with_newline(stdout_head, key_segment, type, payload, stdout_add_newline));
    }
}

struct StandardKeyEmitHeads {
    std::string file_head;
    std::string stdout_default_head;
};

static void emit_standard_key_result(FILE* file,
    const StandardKeyEmitHeads& heads,
    const uint8_t* found_priv_key,
    const uint32_t* found_hash160,
    uint8_t coin_type,
    int64_t round_value,
    bool save)
{
    const uint8_t base_type = endo_base_type_or_self(coin_type);
    bool is_ed = false;
    const char* type = byte_to_coin(base_type, is_ed);
    uint8_t key_to_emit[32];
    fix_endo_key(found_priv_key, key_to_emit, coin_type);
    const std::string key_segment = build_key_segment(key_to_emit, round_value);

    auto emit_default = [&](const char* emit_type, const std::string& payload, bool stdout_add_newline = true) {
        emit_mode_result_line(file, heads.file_head, key_segment, emit_type, payload, heads.stdout_default_head, Silent, stdout_add_newline);
    };

    const uint8_t* payload = reinterpret_cast<const uint8_t*>(found_hash160);
    if (!save) {
        if (found_hash_is_32_bytes(base_type)) {
            emit_default(type, build_hash_payload_words(found_hash160, 8));
        }
        else {
            emit_default(type, build_hash_payload_words(found_hash160, 5), false);
        }
        return;
    }

    switch (base_type) {
    case 0x01:
        emit_default(type, hash160ToBase58_d(payload, 0x00));
        return;
    case 0x02:
    {
        const std::string addr = hash160ToBase58_d(payload, 0x00);
        char output[86]{};
        segwit_addr_encode(output, "bc", 0, payload, 20);
        emit_default(type, addr);
        emit_default("P2WPKH", output);
        return;
    }
    case 0x03:
        emit_default(type, hash160ToBase58_d(payload, 0x05));
        return;
    case 0x04:
    {
        char output[86]{};
        segwit_addr_encode(output, "bc", 1, payload, 32);
        emit_default(type, output);
        return;
    }
    case 0x05:
        emit_default(type, build_hash_payload_words(found_hash160, 5));
        return;
    case 0x06:
        emit_default(type, build_hash_payload_words(found_hash160, 5, true));
        return;
    case 0x20:
    case 0x21:
    case 0x22:
    case 0x52:
    case 0x53:
    case 0x70:
    case 0x71:
        emit_default(type, build_hash_payload_words(found_hash160, 8, true));
        return;
    case 0x30:
    case 0x31:
        emit_default(type, polkadotAddress(payload));
        return;
    case 0x41:
    {
        char addr[64]{};
        filecoin_from_payload(payload, "f1", addr);
        emit_default(type, addr);
        return;
    }
    case 0x42:
    {
        char addr[64]{};
        filecoin_from_payload(payload, "f4", addr);
        emit_default(type, addr);
        return;
    }
    case 0x50:
    {
        char output[86]{};
        iota_bech32(output, "iota", payload, 0x01);
        emit_default(type, output);
        return;
    }
    case 0x51:
    {
        char output[86]{};
        iota_bech32(output, "iota", payload, 0x00);
        emit_default(type, output);
        return;
    }
    case 0x60:
        emit_default(type, encodeBase58_d(payload, 32));
        return;
    case 0x80:
    case 0x81:
    case 0x82:
    case 0x83:
    case 0x84:
    case 0x85:
    case 0x86:
    case 0x87:
    case 0x88:
    case 0x89:
    case 0x8a:
    case 0x8b:
    case 0x8c:
        emit_default(type, generate_ton_address(payload));
        return;
    case 0x90:
    case 0x91:
        emit_default(type, hash160ToXRP(payload, 0x00));
        return;
    case 0x92:
        emit_default(type, tz_from_pkhash(PREFIX_TZ2, payload));
        return;
    case 0x93:
        emit_default(type, tz_from_pkhash(PREFIX_TZ1, payload));
        return;
    default:
        emit_default(type, build_hash_payload_words(found_hash160, found_hash_is_32_bytes(base_type) ? 8 : 5));
        return;
    }
}

static std::string sanitize_brain_visible_text(std::string selected)
{
    std::string out;
    out.reserve(selected.size() * 4u);
    for (unsigned char ch : selected) {
        if (ch < 0x20 || ch == 0x7f) {
            char buf[7];
            std::snprintf(buf, sizeof(buf), "[0x%02X]", ch);
            out += buf;
        }
        else {
            out.push_back(static_cast<char>(ch));
        }
    }
    return out;
}

static std::string build_brain_selected_label(const char* found_string, uint32_t reported_len, uint32_t iter_value)
{
    const uint32_t actual_len = safe_found_string_len(found_string, reported_len);
    std::string selected(found_string, found_string + actual_len);
    selected = sanitize_brain_visible_text(std::move(selected));
    selected += ":hex(";
    selected += hex_bytes(reinterpret_cast<const uint8_t*>(found_string), actual_len);
    selected += "):ITERATION ";
    selected += std::to_string(iter_value);
    return selected;
}

static std::string build_mode_selected_head(const char* mode, const std::string& selected, bool stdout_head)
{
    std::string head;
    if (stdout_head) {
        head = "\n[!] Found: ";
    }
    head += mode;
    head.push_back(':');
    head += selected;
    head.push_back(':');
    return head;
}

static void emit_brain_like_result(FILE* file,
    const std::string& selected,
    const uint8_t* found_priv_key,
    const uint32_t* found_hash160,
    uint8_t coin_type,
    int64_t round_value,
    bool save)
{
    const StandardKeyEmitHeads heads{
        build_mode_selected_head("Brain", selected, false),
        build_mode_selected_head("Brain", selected, true)
    };
    emit_standard_key_result(file, heads, found_priv_key, found_hash160, coin_type, round_value, save);
}

static void reset_symbol_buffer(void* ptr, size_t bytes)
{
    if (ptr != nullptr && bytes > 0) {
        cudaMemset(ptr, 0, bytes);
    }
}

static void reset_found_buffers(uint32_t count)
{
    void* counter = nullptr;
    if (cudaGetSymbolAddress(&counter, d_resultsCount) == cudaSuccess) {
        cudaMemset(counter, 0, sizeof(unsigned long long));
    }

    char (*dev_strings)[512] = nullptr;
    unsigned char (*dev_priv)[64] = nullptr;
    uint32_t (*dev_hash)[20] = nullptr;
    uint32_t (*dev_len)[1] = nullptr;
    uint32_t (*dev_iter)[1] = nullptr;
    uint8_t* dev_type = nullptr;
    int64_t* dev_round = nullptr;
    cudaMemcpyFromSymbol(&dev_strings, d_foundStrings, sizeof(dev_strings));
    cudaMemcpyFromSymbol(&dev_priv, d_foundPrvKeys, sizeof(dev_priv));
    cudaMemcpyFromSymbol(&dev_hash, d_foundHash160, sizeof(dev_hash));
    cudaMemcpyFromSymbol(&dev_len, d_len, sizeof(dev_len));
    cudaMemcpyFromSymbol(&dev_iter, d_iter, sizeof(dev_iter));
    cudaMemcpyFromSymbol(&dev_type, d_type, sizeof(dev_type));
    cudaMemcpyFromSymbol(&dev_round, d_round, sizeof(dev_round));

    reset_symbol_buffer(dev_strings, static_cast<size_t>(count) * 512);
    reset_symbol_buffer(dev_priv, static_cast<size_t>(count) * 64);
    reset_symbol_buffer(dev_hash, static_cast<size_t>(count) * 20 * sizeof(uint32_t));
    reset_symbol_buffer(dev_len, static_cast<size_t>(count) * sizeof(uint32_t));
    reset_symbol_buffer(dev_iter, static_cast<size_t>(count) * sizeof(uint32_t));
    reset_symbol_buffer(dev_type, static_cast<size_t>(count));
    reset_symbol_buffer(dev_round, static_cast<size_t>(count) * sizeof(int64_t));
}

static uint32_t snapshot_count()
{
    unsigned long long results = 0;
    cudaMemcpyFromSymbol(&results, d_resultsCount, sizeof(results));
    if (results > MAX_FOUNDS) {
        return MAX_FOUNDS;
    }
    return static_cast<uint32_t>(results);
}

static void SaveResultPRIV_Worker(FILE* file,
    uint32_t* pFounds,
    bool save,
    unsigned char (*foundPrvKeys)[64],
    uint32_t (*foundHash160)[20],
    uint8_t* coin_type,
    int64_t* rounds,
    unsigned long long count)
{
    uint32_t emitted = 0;
    const StandardKeyEmitHeads heads{ "", "\n[!] Found: " };

    for (uint64_t i = 0; i < count; i++) {
        if (!cpu_postcheck(foundHash160[i])) {
            false_positive.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        ++emitted;
        emit_standard_key_result(file, heads, foundPrvKeys[i], foundHash160[i], coin_type[i], rounds[i], save);
        save_output_flush(file);
    }

    if (pFounds != nullptr && emitted != 0) {
        std::lock_guard<std::mutex> lock(g_found_mutex);
        *pFounds += emitted;
    }
    if (emitted != 0) {
        g_runtime_founds.fetch_add(emitted, std::memory_order_relaxed);
    }

    delete[] foundPrvKeys;
    delete[] foundHash160;
    delete[] coin_type;
    delete[] rounds;
    STOP_THREAD = false;
}

static void SaveResultBrain_Worker(FILE* file,
    uint32_t* pFounds,
    bool save,
    char (*foundStrings)[512],
    unsigned char (*foundPrvKeys)[64],
    uint32_t (*foundHash160)[20],
    uint8_t* coin_type,
    uint32_t (*len_h)[1],
    uint32_t (*iter_h)[1],
    int64_t* rounds,
    unsigned long long count)
{
    uint32_t emitted = 0;

    for (uint64_t i = 0; i < count; i++) {
        if (!cpu_postcheck(foundHash160[i])) {
            false_positive.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        ++emitted;
        const std::string selected = build_brain_selected_label(&foundStrings[i][0], len_h[i][0], iter_h[i][0]);
        emit_brain_like_result(file, selected, foundPrvKeys[i], foundHash160[i], coin_type[i], rounds[i], save);
        save_output_flush(file);
    }

    if (pFounds != nullptr && emitted != 0) {
        std::lock_guard<std::mutex> lock(g_found_mutex);
        *pFounds += emitted;
    }
    if (emitted != 0) {
        g_runtime_founds.fetch_add(emitted, std::memory_order_relaxed);
    }

    delete[] foundStrings;
    delete[] foundPrvKeys;
    delete[] foundHash160;
    delete[] coin_type;
    delete[] len_h;
    delete[] iter_h;
    delete[] rounds;
    STOP_THREAD = false;
}

__host__ void SaveResultPRIV(FILE* file, uint32_t& Founds, bool save, vector<string>)
{
    STOP_THREAD = true;
    const uint32_t count = snapshot_count();
    if (count == 0) {
        reset_found_buffers(0);
        STOP_THREAD = false;
        return;
    }

    unsigned char (*dev_priv)[64] = nullptr;
    uint32_t (*dev_hash)[20] = nullptr;
    uint8_t* dev_type = nullptr;
    int64_t* dev_round = nullptr;
    cudaMemcpyFromSymbol(&dev_priv, d_foundPrvKeys, sizeof(dev_priv));
    cudaMemcpyFromSymbol(&dev_hash, d_foundHash160, sizeof(dev_hash));
    cudaMemcpyFromSymbol(&dev_type, d_type, sizeof(dev_type));
    cudaMemcpyFromSymbol(&dev_round, d_round, sizeof(dev_round));

    unsigned char (*foundPrvKeys)[64] = new unsigned char[count][64];
    uint32_t (*foundHash160)[20] = new uint32_t[count][20];
    uint8_t* coin_type = new uint8_t[count];
    int64_t* rounds = new int64_t[count];

    cudaMemcpy(foundPrvKeys, dev_priv, static_cast<size_t>(count) * 64, cudaMemcpyDeviceToHost);
    cudaMemcpy(foundHash160, dev_hash, static_cast<size_t>(count) * 20 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(coin_type, dev_type, static_cast<size_t>(count), cudaMemcpyDeviceToHost);
    cudaMemcpy(rounds, dev_round, static_cast<size_t>(count) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    reset_found_buffers(count);

    {
        std::lock_guard<std::mutex> lock(g_save_threads_mutex);
        enqueue_async_save_task_for_current_gpu("priv", count, save, SaveResultPRIV_Worker, file, &Founds, save, foundPrvKeys, foundHash160, coin_type, rounds, static_cast<unsigned long long>(count));
    }
}

__host__ void SaveResultBrain(FILE* file, uint32_t& Founds, bool save, vector<string>)
{
    STOP_THREAD = true;
    const uint32_t count = snapshot_count();
    if (count == 0) {
        reset_found_buffers(0);
        STOP_THREAD = false;
        return;
    }

    char (*dev_strings)[512] = nullptr;
    unsigned char (*dev_priv)[64] = nullptr;
    uint32_t (*dev_hash)[20] = nullptr;
    uint8_t* dev_type = nullptr;
    uint32_t (*dev_len)[1] = nullptr;
    uint32_t (*dev_iter)[1] = nullptr;
    int64_t* dev_round = nullptr;
    cudaMemcpyFromSymbol(&dev_strings, d_foundStrings, sizeof(dev_strings));
    cudaMemcpyFromSymbol(&dev_priv, d_foundPrvKeys, sizeof(dev_priv));
    cudaMemcpyFromSymbol(&dev_hash, d_foundHash160, sizeof(dev_hash));
    cudaMemcpyFromSymbol(&dev_type, d_type, sizeof(dev_type));
    cudaMemcpyFromSymbol(&dev_len, d_len, sizeof(dev_len));
    cudaMemcpyFromSymbol(&dev_iter, d_iter, sizeof(dev_iter));
    cudaMemcpyFromSymbol(&dev_round, d_round, sizeof(dev_round));

    char (*foundStrings)[512] = new char[count][512];
    unsigned char (*foundPrvKeys)[64] = new unsigned char[count][64];
    uint32_t (*foundHash160)[20] = new uint32_t[count][20];
    uint8_t* coin_type = new uint8_t[count];
    uint32_t (*len_h)[1] = new uint32_t[count][1];
    uint32_t (*iter_h)[1] = new uint32_t[count][1];
    int64_t* rounds = new int64_t[count];

    cudaMemcpy(foundStrings, dev_strings, static_cast<size_t>(count) * 512, cudaMemcpyDeviceToHost);
    cudaMemcpy(foundPrvKeys, dev_priv, static_cast<size_t>(count) * 64, cudaMemcpyDeviceToHost);
    cudaMemcpy(foundHash160, dev_hash, static_cast<size_t>(count) * 20 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(coin_type, dev_type, static_cast<size_t>(count), cudaMemcpyDeviceToHost);
    cudaMemcpy(len_h, dev_len, static_cast<size_t>(count) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(iter_h, dev_iter, static_cast<size_t>(count) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(rounds, dev_round, static_cast<size_t>(count) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    reset_found_buffers(count);

    {
        std::lock_guard<std::mutex> lock(g_save_threads_mutex);
        enqueue_async_save_task_for_current_gpu("brain", count, save, SaveResultBrain_Worker, file, &Founds, save, foundStrings, foundPrvKeys, foundHash160, coin_type, len_h, iter_h, rounds, static_cast<unsigned long long>(count));
    }
}
