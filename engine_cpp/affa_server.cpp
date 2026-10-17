// Affan AI Engine Studio - embedded local HTTP server.  (c) MABA Studios
#include "affa_server.h"
#include "ui_assets.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <map>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <commdlg.h>
typedef SOCKET sock_t;
#define CLOSESOCK closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
typedef int sock_t;
#define CLOSESOCK ::close
#define INVALID_SOCKET (-1)
#endif

namespace fs = std::filesystem;

namespace {

// ------------------------------------------------------------------ JSON ---
struct J {
    enum T { N, B, Num, S, A, O } t = N; bool b = false; double n = 0; std::string s;
    std::vector<J> a; std::vector<std::pair<std::string, J>> o;
    const J* get(const char* k) const { for (auto& p : o) if (p.first == k) return &p.second; return nullptr; }
    std::string str(const char* k, const std::string& d = "") const { auto* v = get(k); return v && v->t == S ? v->s : d; }
    double num(const char* k, double d) const { auto* v = get(k); return v && v->t == Num ? v->n : d; }
};
struct JP {
    const std::string& s; size_t i = 0; bool ok = true;
    explicit JP(const std::string& str) : s(str) {}
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) i++; }
    static void put_utf8(std::string& o, unsigned cp) {
        if (cp < 0x80) o += (char)cp;
        else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
        else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
    }
    std::string str() {
        std::string o; i++;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char c = s[++i];
                switch (c) {
                    case 'n': o += '\n'; break; case 't': o += '\t'; break; case 'r': o += '\r'; break;
                    case 'b': o += '\b'; break; case 'f': o += '\f'; break;
                    case 'u': {
                        if (i + 4 >= s.size()) { ok = false; return o; }
                        unsigned cp = (unsigned)std::stoul(s.substr(i + 1, 4), nullptr, 16); i += 4;
                        if (cp >= 0xD800 && cp < 0xDC00 && i + 6 < s.size() && s[i + 1] == '\\' && s[i + 2] == 'u') {
                            unsigned lo = (unsigned)std::stoul(s.substr(i + 3, 4), nullptr, 16);
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6;
                        }
                        put_utf8(o, cp); break;
                    }
                    default: o += c;
                }
                i++;
            } else o += s[i++];
        }
        if (i >= s.size()) ok = false; i++;
        return o;
    }
    J val() {
        J v; ws(); if (i >= s.size()) { ok = false; return v; }
        char c = s[i];
        if (c == '{') {
            v.t = J::O; i++; ws();
            if (s[i] == '}') { i++; return v; }
            while (ok) {
                ws(); if (s[i] != '"') { ok = false; break; }
                std::string k = str(); ws(); if (s[i] != ':') { ok = false; break; } i++;
                v.o.emplace_back(k, val()); ws();
                if (s[i] == ',') { i++; continue; } if (s[i] == '}') { i++; break; } ok = false;
            }
        } else if (c == '[') {
            v.t = J::A; i++; ws();
            if (s[i] == ']') { i++; return v; }
            while (ok) { v.a.push_back(val()); ws(); if (s[i] == ',') { i++; continue; } if (s[i] == ']') { i++; break; } ok = false; }
        } else if (c == '"') { v.t = J::S; v.s = str(); }
        else if (!s.compare(i, 4, "true")) { v.t = J::B; v.b = true; i += 4; }
        else if (!s.compare(i, 5, "false")) { v.t = J::B; i += 5; }
        else if (!s.compare(i, 4, "null")) { i += 4; }
        else { char* e; v.n = std::strtod(s.c_str() + i, &e); if (e == s.c_str() + i) ok = false; i = e - s.c_str(); v.t = J::Num; }
        return v;
    }
};
std::string jesc(const std::string& s) {
    std::string o; char b[8];
    for (unsigned char c : s) {
        if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t"; else if (c < 0x20) { std::snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char)c;
    }
    return o;
}

// ----------------------------------------------------------------- state ---
AffaServerConfig g_cfg;
std::mutex g_mu;                                  // guards g_model / g_path
std::shared_ptr<AffaModel> g_model;
std::string g_path, g_display;
std::atomic<long long> g_last_ping{-1};
std::atomic<bool> g_run{false};
sock_t g_listen = INVALID_SOCKET;
std::thread g_thread;
long long now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::shared_ptr<AffaModel> cur_model() { std::lock_guard<std::mutex> lk(g_mu); return g_model; }

std::string models_json() {
    std::string o = "[";
    bool first = true;
    for (const std::string& d : {g_cfg.models_dir, g_cfg.extra_models_dir}) {
        if (d.empty()) continue;
        std::error_code ec;
        for (auto& e : fs::directory_iterator(fs::u8path(d), ec)) {
            if (ec) break;
            if (!e.is_regular_file(ec) || e.path().extension() != ".affa") continue;
            if (!first) o += ","; first = false;
            o += "{\"name\":\"" + jesc(e.path().filename().u8string()) + "\",\"path\":\"" + jesc(e.path().u8string()) +
                 "\",\"mb\":" + std::to_string((long long)(e.file_size(ec) / 1048576)) + "}";
        }
    }
    return o + "]";
}
std::string state_json() {
    std::shared_ptr<AffaModel> m = cur_model(); std::string path, disp;
    { std::lock_guard<std::mutex> lk(g_mu); path = g_path; disp = g_display; }
    std::string o = "{\"platform\":\"" + g_cfg.platform + "\",\"loaded\":" + (m ? "true" : "false") + ",";
    if (m) {
        const char* g = affa_gpu_device(m.get());
        if (g && *g) o += "\"gpu\":\"" + jesc(g) + "\",";
        o += "\"info\":" + std::string(affa_info_json(m.get())) + ",\"path\":\"" + jesc(path) + "\",\"display\":\"" + jesc(disp) + "\",";
    }
    o += "\"models_dir\":\"" + jesc(g_cfg.models_dir) + "\",\"default_threads\":" + std::to_string(g_cfg.default_threads) +
         ",\"cores\":" + std::to_string(std::thread::hardware_concurrency()) + ",\"native_dialog\":" + (g_cfg.native_file_dialog ? "true" : "false") +
         ",\"models\":" + models_json() + "}";
    return o;
}

// ------------------------------------------------------------------ http ---
bool send_all(sock_t s, const char* p, size_t n) {
    while (n) {
        int k = (int)send(s, p, (int)(n > (1 << 20) ? (1 << 20) : n), 0);
        if (k <= 0) return false; p += k; n -= (size_t)k;
    }
    return true;
}
void respond(sock_t s, int code, const char* ctype, const std::string& body, const char* extra = "") {
    const char* st = code == 200 ? "OK" : code == 400 ? "Bad Request" : code == 404 ? "Not Found" : "Error";
    char h[512];
    int n = std::snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nCache-Control: no-store\r\nConnection: close\r\n%s\r\n",
                          code, st, ctype, body.size(), extra);
    send_all(s, h, (size_t)n); send_all(s, body.data(), body.size());
}
void respond_json(sock_t s, int code, const std::string& b) { respond(s, code, "application/json; charset=utf-8", b); }
bool send_chunk(sock_t s, const std::string& d) {
    char h[32]; int n = std::snprintf(h, sizeof h, "%zx\r\n", d.size());
    return send_all(s, h, (size_t)n) && send_all(s, d.data(), d.size()) && send_all(s, "\r\n", 2);
}

struct Req { std::string method, path, query, body; std::map<std::string, std::string> hdr; size_t content_length = 0; std::string leftover; };

bool read_head(sock_t s, Req& r) {
    std::string buf; char tmp[8192];
    while (buf.find("\r\n\r\n") == std::string::npos) {
        int k = (int)recv(s, tmp, sizeof tmp, 0); if (k <= 0) return false; buf.append(tmp, (size_t)k);
        if (buf.size() > 65536) return false;
    }
    size_t he = buf.find("\r\n\r\n");
    std::string head = buf.substr(0, he); r.leftover = buf.substr(he + 4);
    size_t le = head.find("\r\n"); std::string line = head.substr(0, le);
    size_t a = line.find(' '), b = line.find(' ', a + 1);
    if (a == std::string::npos || b == std::string::npos) return false;
    r.method = line.substr(0, a); std::string target = line.substr(a + 1, b - a - 1);
    size_t q = target.find('?'); r.path = target.substr(0, q); if (q != std::string::npos) r.query = target.substr(q + 1);
    size_t p = le;
    while (p != std::string::npos && p < head.size()) {
        p += 2; size_t e = head.find("\r\n", p); std::string h = head.substr(p, e == std::string::npos ? std::string::npos : e - p);
        size_t c = h.find(':');
        if (c != std::string::npos) {
            std::string k = h.substr(0, c); for (auto& ch : k) ch = (char)std::tolower((unsigned char)ch);
            size_t v = c + 1; while (v < h.size() && h[v] == ' ') v++; r.hdr[k] = h.substr(v);
        }
        p = e;
    }
    auto it = r.hdr.find("content-length"); if (it != r.hdr.end()) r.content_length = (size_t)std::strtoull(it->second.c_str(), nullptr, 10);
    return true;
}
bool read_body(sock_t s, Req& r, size_t cap) {
    if (r.content_length > cap) return false;
    r.body = r.leftover; char tmp[16384];
    while (r.body.size() < r.content_length) { int k = (int)recv(s, tmp, sizeof tmp, 0); if (k <= 0) return false; r.body.append(tmp, (size_t)k); }
    return true;
}
std::string qparam(const std::string& q, const std::string& key) {
    size_t p = 0;
    while (p < q.size()) {
        size_t e = q.find('&', p); std::string kv = q.substr(p, e == std::string::npos ? std::string::npos : e - p);
        size_t eq = kv.find('=');
        if (kv.substr(0, eq) == key && eq != std::string::npos) {
            std::string v = kv.substr(eq + 1), o;
            for (size_t i = 0; i < v.size(); i++) { if (v[i] == '%' && i + 2 < v.size()) { o += (char)std::stoi(v.substr(i + 1, 2), nullptr, 16); i += 2; } else if (v[i] == '+') o += ' '; else o += v[i]; }
            return o;
        }
        if (e == std::string::npos) break; p = e + 1;
    }
    return "";
}

struct ChatCtx { sock_t s; bool alive = true; };
int on_token(const char* t, int n, void* u) {
    ChatCtx* c = (ChatCtx*)u;
    if (!c->alive) return 1;
    c->alive = send_chunk(c->s, "{\"t\":\"" + jesc(std::string(t, (size_t)n)) + "\"}\n");
    return c->alive ? 0 : 1;
}

bool set_model(std::shared_ptr<AffaModel> m, const std::string& path, const std::string& display) {
    std::lock_guard<std::mutex> lk(g_mu); g_model = m; g_path = path; g_display = display; return true;
}

void handle(sock_t s) {
    Req r;
    if (!read_head(s, r)) { CLOSESOCK(s); return; }
    try {
        if (r.method == "GET" && (r.path == "/" || r.path == "/index.html")) respond(s, 200, "text/html; charset=utf-8", std::string((const char*)kIndexHtml, kIndexHtmlLen));
        else if (r.method == "GET" && (r.path == "/logo.png" || r.path == "/favicon.ico" || r.path == "/apple-touch-icon.png"))
            respond(s, 200, "image/png", std::string((const char*)kLogoPng, kLogoPngLen), "Cache-Control: max-age=86400\r\n");
        else if (r.path == "/api/ping") { g_last_ping = now_ms(); respond_json(s, 200, "{\"ok\":true}"); }
        else if (r.path == "/api/state") respond_json(s, 200, state_json());
        else if (r.path == "/api/pick" && g_cfg.native_file_dialog) {
#if defined(_WIN32)
            wchar_t file[2048] = {0}; OPENFILENAMEW ofn = {sizeof ofn};
            ofn.hwndOwner = GetForegroundWindow(); ofn.lpstrFile = file; ofn.nMaxFile = 2048;
            ofn.lpstrFilter = L"Affan AI model (*.affa)\0*.affa\0All files\0*.*\0"; ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
            ofn.lpstrTitle = L"Select an .affa model";
            std::string out;
            if (GetOpenFileNameW(&ofn)) { int n = WideCharToMultiByte(CP_UTF8, 0, file, -1, nullptr, 0, nullptr, nullptr); out.resize(n - 1); WideCharToMultiByte(CP_UTF8, 0, file, -1, &out[0], n, nullptr, nullptr); }
            respond_json(s, 200, "{\"path\":\"" + jesc(out) + "\"}");
#endif
        }
        else if (r.method == "POST" && r.path == "/api/load") {
            if (!read_body(s, r, 1 << 20)) { respond_json(s, 400, "{\"error\":\"bad body\"}"); CLOSESOCK(s); return; }
            JP p(r.body); J j = p.val(); std::string path = j.str("path");
            int th = (int)j.num("threads", g_cfg.default_threads), ctx = (int)j.num("ctx", g_cfg.default_ctx);
            std::string err;
            if (!affa_server_load_path(path, err)) { respond_json(s, 200, "{\"error\":\"" + jesc(err) + "\"}"); }
            else { (void)th; (void)ctx; respond_json(s, 200, state_json()); }
        }
        else if (r.method == "POST" && r.path == "/api/unload") { set_model(nullptr, "", ""); respond_json(s, 200, state_json()); }
        else if (r.method == "POST" && r.path == "/api/upload") {
            std::string name = fs::path(fs::u8path(qparam(r.query, "name"))).filename().u8string();
            if (name.empty() || fs::path(fs::u8path(name)).extension() != ".affa") { respond_json(s, 400, "{\"error\":\"file must end with .affa\"}"); CLOSESOCK(s); return; }
            std::error_code ec; fs::create_directories(fs::u8path(g_cfg.models_dir), ec);
            fs::path dst = fs::u8path(g_cfg.models_dir) / fs::u8path(name), part = dst; part += ".part";
            FILE* f = fopen(part.string().c_str(), "wb");
            if (!f) { respond_json(s, 200, "{\"error\":\"cannot write to models folder\"}"); CLOSESOCK(s); return; }
            size_t got = r.leftover.size(); fwrite(r.leftover.data(), 1, got, f);
            std::vector<char> buf(1 << 20);
            while (got < r.content_length) { int k = (int)recv(s, buf.data(), (int)std::min<size_t>(buf.size(), r.content_length - got), 0); if (k <= 0) break; fwrite(buf.data(), 1, (size_t)k, f); got += (size_t)k; }
            fclose(f);
            if (got != r.content_length) { fs::remove(part, ec); respond_json(s, 200, "{\"error\":\"upload interrupted\"}"); }
            else { fs::rename(part, dst, ec); std::string err;
                   if (!affa_server_load_path(dst.u8string(), err)) respond_json(s, 200, "{\"error\":\"" + jesc(err) + "\"}"); else respond_json(s, 200, state_json()); }
        }
        else if (r.method == "POST" && r.path == "/api/stop") { auto m = cur_model(); if (m) affa_stop(m.get()); respond_json(s, 200, "{\"ok\":true}"); }
        else if (r.method == "POST" && r.path == "/api/backend") {
            if (!read_body(s, r, 1 << 16)) { respond_json(s, 400, "{\"error\":\"bad body\"}"); CLOSESOCK(s); return; }
            auto m = cur_model();
            if (!m) { respond_json(s, 200, "{\"error\":\"no model loaded\"}"); CLOSESOCK(s); return; }
            JP p(r.body); J j = p.val(); std::string mode = j.str("backend", "auto");
            if (affa_set_backend(m.get(), mode.c_str()) != 0) {
                respond_json(s, 200, "{\"error\":\"Requested backend is not available on this hardware.\"}");
            } else {
                respond_json(s, 200, state_json());
            }
        }
        else if (r.method == "POST" && r.path == "/api/bench") {
            auto m = cur_model(); if (!m) { respond_json(s, 200, "{\"error\":\"no model loaded\"}"); }
            else { double o[3]; affa_bench(m.get(), 8, 4, o); affa_bench(m.get(), 64, 48, o);
                   char b[200]; std::snprintf(b, sizeof b, "{\"prefill_tps\":%.1f,\"decode_tps\":%.2f,\"prefill_ms\":%.1f}", o[0], o[1], o[2]); respond_json(s, 200, b); }
        }
        else if (r.method == "POST" && r.path == "/api/chat") {
            if (!read_body(s, r, 8 << 20)) { respond_json(s, 400, "{\"error\":\"bad body\"}"); CLOSESOCK(s); return; }
            auto m = cur_model();
            if (!m) { respond_json(s, 200, "{\"error\":\"No model loaded. Open Settings and choose an .affa model.\"}"); CLOSESOCK(s); return; }
            JP p(r.body); J j = p.val();
            const J* ms = j.get("messages");
            if (!p.ok || !ms || ms->t != J::A || ms->a.empty()) { respond_json(s, 400, "{\"error\":\"bad request\"}"); CLOSESOCK(s); return; }
            std::vector<std::string> roles, cont;
            for (auto& x : ms->a) { roles.push_back(x.str("role", "user")); cont.push_back(x.str("content")); }
            std::vector<const char*> rp, cp; for (size_t i = 0; i < roles.size(); i++) { rp.push_back(roles[i].c_str()); cp.push_back(cont[i].c_str()); }
            AffaParams P; affa_default_params(m.get(), &P);
            P.temperature = (float)j.num("temperature", P.temperature); P.top_p = (float)j.num("top_p", P.top_p);
            P.top_k = (int)j.num("top_k", P.top_k); P.rep_penalty = (float)j.num("rep_penalty", P.rep_penalty);
            P.max_new_tokens = (int)j.num("max_tokens", 2048); P.seed = (int)j.num("seed", 0);
            const J* thk = j.get("think");
            if (thk) {
                if (thk->t == J::B) P.think = thk->b ? 1 : 0;
                else if (thk->t == J::Num) P.think = thk->n > 0 ? 1 : 0;
                else if (thk->t == J::S) P.think = (thk->s == "true" || thk->s == "1") ? 1 : 0;
            }
            int th = (int)j.num("threads", 0); if (th > 0) affa_set_threads(m.get(), th);
            bool has_sys = j.get("system") != nullptr; std::string sys = j.str("system");
            const char* hdr = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson; charset=utf-8\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
            send_all(s, hdr, strlen(hdr));
            ChatCtx cc{s}; AffaStats st; std::memset(&st, 0, sizeof st);
            int n = affa_chat(m.get(), (int)rp.size(), rp.data(), cp.data(), has_sys ? sys.c_str() : nullptr, &P, on_token, &cc, &st);
            char b[400];
            if (n < 0) std::snprintf(b, sizeof b, "{\"error\":\"conversation too long for the context window\"}\n");
            else std::snprintf(b, sizeof b, "{\"done\":true,\"stats\":{\"prompt_tokens\":%d,\"reused\":%d,\"gen_tokens\":%d,\"ttft_ms\":%.0f,\"tps\":%.2f,\"decode_ms\":%.0f}}\n",
                               st.prompt_tokens, st.reused_tokens, st.gen_tokens, st.ttft_ms, st.tps, st.decode_ms);
            send_chunk(s, b); send_all(s, "0\r\n\r\n", 5);
        }
        else respond(s, 404, "text/plain", "not found");
    } catch (const std::exception& e) {
        respond_json(s, 500, std::string("{\"error\":\"") + jesc(e.what()) + "\"}");
    }
    CLOSESOCK(s);
}

void accept_loop() {
    while (g_run) {
        sockaddr_in ca; socklen_t cl = sizeof ca;
        sock_t c = accept(g_listen, (sockaddr*)&ca, &cl);
        if (c == INVALID_SOCKET) { if (!g_run) break; continue; }
        int one = 1; setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
        std::thread(handle, c).detach();
    }
}

}  // namespace

bool affa_server_load_path(const std::string& path, std::string& err) {
    char e[512] = {0};
    AffaModel* m = affa_load_file(path.c_str(), g_cfg.default_threads, g_cfg.default_ctx, e, sizeof e);
    if (!m) { err = e; return false; }
    std::string disp = fs::u8path(path).filename().u8string();
    set_model(std::shared_ptr<AffaModel>(m, affa_free), path, disp);
    return true;
}
bool affa_server_load_fd(int fd, const std::string& display_name, std::string& err) {
    char e[512] = {0};
    AffaModel* m = affa_load_fd(fd, g_cfg.default_threads, g_cfg.default_ctx, e, sizeof e);
    if (!m) { err = e; return false; }
    set_model(std::shared_ptr<AffaModel>(m, affa_free), display_name, display_name);
    return true;
}
long long affa_server_ms_since_ping() { long long p = g_last_ping; return p < 0 ? -1 : now_ms() - p; }

int affa_server_start(const AffaServerConfig& cfg, int preferred_port) {
    g_cfg = cfg;
#if defined(_WIN32)
    WSADATA w; WSAStartup(MAKEWORD(2, 2), &w);
#endif
    g_listen = socket(AF_INET, SOCK_STREAM, 0);
    if (g_listen == INVALID_SOCKET) return -1;
    int one = 1; setsockopt(g_listen, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof one);
    int port = -1;
    for (int p = preferred_port; p < preferred_port + 50; p++) {
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((unsigned short)p); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(g_listen, (sockaddr*)&a, sizeof a) == 0) { port = p; break; }
    }
    if (port < 0 || listen(g_listen, 16) != 0) return -1;
    g_run = true; g_thread = std::thread(accept_loop);
    return port;
}
void affa_server_stop() {
    g_run = false;
    if (g_listen != INVALID_SOCKET) { CLOSESOCK(g_listen); g_listen = INVALID_SOCKET; }
    if (g_thread.joinable()) g_thread.detach();
    set_model(nullptr, "", "");
}
