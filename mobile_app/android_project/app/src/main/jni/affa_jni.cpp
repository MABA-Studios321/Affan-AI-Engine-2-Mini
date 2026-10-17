// Android JNI Bridge for Affan AI Engine
#include "affa_server.h"
#include <jni.h>
#include <string>

extern "C" {

JNIEXPORT jint JNICALL
Java_com_maba_affanai_MainActivity_nativeStartServer(JNIEnv* env, jobject thiz, jstring modelsDir, jint threads) {
    const char* dir = env->GetStringUTFChars(modelsDir, nullptr);
    AffaServerConfig cfg;
    cfg.platform = "android";
    cfg.models_dir = dir ? dir : "";
    cfg.default_threads = 0; // 0 = auto-detect big.LITTLE CPU topology (reserve 1 efficient core for OS)
    cfg.default_ctx = 2048;
    cfg.native_file_dialog = false;

    int port = affa_server_start(cfg, 37821);
    if (dir && port > 0) {
        std::string defaultPath = std::string(dir) + "/affan_ai_engine_2_mini.affa";
        std::string err;
        affa_server_load_path(defaultPath, err);
    }
    if (dir) env->ReleaseStringUTFChars(modelsDir, dir);
    return port;
}

JNIEXPORT void JNICALL
Java_com_maba_affanai_MainActivity_nativeStopServer(JNIEnv* env, jobject thiz) {
    affa_server_stop();
}

JNIEXPORT jboolean JNICALL
Java_com_maba_affanai_MainActivity_nativeLoadFd(JNIEnv* env, jobject thiz, jint fd, jstring displayName) {
    const char* name = env->GetStringUTFChars(displayName, nullptr);
    std::string err;
    bool ok = affa_server_load_fd(fd, name ? name : "model.affa", err);
    if (name) env->ReleaseStringUTFChars(displayName, name);
    return ok ? JNI_TRUE : JNI_FALSE;
}

}
