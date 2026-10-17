package com.maba.affanai;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import android.webkit.JavascriptInterface;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;

public class MainActivity extends Activity {
    static {
        System.loadLibrary("affa_engine");
    }

    private native int nativeStartServer(String modelsDir, int threads);
    private native void nativeStopServer();
    private native boolean nativeLoadFd(int fd, String displayName);

    private WebView webView;
    private int serverPort = -1;
    private static final int REQ_PICK_FOLDER = 1001;
    private static final int REQ_PICK_FILE = 1002;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        webView = new WebView(this);
        webView.setBackgroundColor(0xFF0E0F11);
        setContentView(webView);

        WebSettings ws = webView.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true);
        ws.setAllowFileAccess(true);
        ws.setCacheMode(WebSettings.LOAD_NO_CACHE);

        webView.addJavascriptInterface(new AndroidBridge(), "AffaAndroid");
        webView.setWebViewClient(new WebViewClient());
        webView.loadData("<body style='background:#0e0f11;color:#8c8f98;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>Preparing Affan AI Engine…</body>", "text/html", "utf-8");

        final File modelsDir = new File(getFilesDir(), "models");
        modelsDir.mkdirs();
        final int threads = Math.max(2, Runtime.getRuntime().availableProcessors() - 1);
        new Thread(new Runnable() {
            @Override
            public void run() {
                installBundledModel(modelsDir);
                serverPort = nativeStartServer(modelsDir.getAbsolutePath(), threads);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (serverPort > 0) webView.loadUrl("http://127.0.0.1:" + serverPort + "/");
                    }
                });
            }
        }).start();
    }

    // Extract assets/affan_ai_engine_2_mini.affa -> files/models/ once (re-copies if size differs).
    private void installBundledModel(File modelsDir) {
        final String name = "affan_ai_engine_2_mini.affa";
        File dst = new File(modelsDir, name);
        try {
            android.content.res.AssetFileDescriptor afd = null;
            long want = -1;
            try {
                afd = getAssets().openFd(name);
                want = afd.getLength();
                afd.close();
            } catch (Exception e) { /* compressed asset: size unknown, fall back to marker check */ }
            if (dst.exists() && (want < 0 ? dst.length() > 0 : dst.length() == want)) return;
            File part = new File(modelsDir, name + ".part");
            java.io.InputStream in = getAssets().open(name);
            java.io.FileOutputStream out = new java.io.FileOutputStream(part);
            byte[] buf = new byte[1 << 20];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
            out.getFD().sync();
            out.close();
            in.close();
            if (dst.exists()) dst.delete();
            part.renameTo(dst);
        } catch (Exception e) {
            android.util.Log.e("AffaEngine", "model install failed: " + e);
        }
    }

    public class AndroidBridge {
        @JavascriptInterface
        public void pickFolder() {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
            startActivityForResult(intent, REQ_PICK_FOLDER);
        }

        @JavascriptInterface
        public void pickFile() {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("*/*");
            startActivityForResult(intent, REQ_PICK_FILE);
        }

        @JavascriptInterface
        public void loadUri(String uriStr, String name) {
            try {
                Uri uri = Uri.parse(uriStr);
                ParcelFileDescriptor pfd = getContentResolver().openFileDescriptor(uri, "r");
                if (pfd != null) {
                    int fd = pfd.detachFd();
                    boolean ok = nativeLoadFd(fd, name);
                    sendJs("loaded", ok ? "{\"ok\":true}" : "{\"error\":\"Failed to load model file\"}");
                } else {
                    sendJs("loaded", "{\"error\":\"Cannot open file descriptor\"}");
                }
            } catch (Exception e) {
                sendJs("loaded", "{\"error\":\"" + e.getMessage() + "\"}");
            }
        }

        @JavascriptInterface
        public void requestFolderList() {
            // Echo current state
        }
    }

    private void sendJs(final String type, final String json) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                webView.evaluateJavascript("window.onAndroid && window.onAndroid('" + type + "', '" + json.replace("'", "\\'") + "');", null);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null) return;

        if (requestCode == REQ_PICK_FILE) {
            Uri uri = data.getData();
            if (uri != null) {
                String name = "model.affa";
                Cursor c = getContentResolver().query(uri, null, null, null, null);
                if (c != null && c.moveToFirst()) {
                    int nameIdx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                    if (nameIdx >= 0) name = c.getString(nameIdx);
                    c.close();
                }
                new AndroidBridge().loadUri(uri.toString(), name);
            }
        }
    }

    @Override
    protected void onDestroy() {
        nativeStopServer();
        super.onDestroy();
    }
}
