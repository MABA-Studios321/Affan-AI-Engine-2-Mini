import os
import zipfile

def build_release_zip():
    out_zip = r"D:\affan_ai_engine_2_mini\Affan_AI_Engine_2_Mini.zip"
    root = "Affan AI Engine 2 Mini"
    files = [
        (r"D:\affan_ai_engine_2_mini\dist\Affan AI Engine 2 Mini.exe", f"{root}/Affan AI Engine 2 Mini.exe"),
        (r"D:\affan_ai_engine_2_mini\dist\models\affan_ai_engine_2_mini.affa", f"{root}/models/affan_ai_engine_2_mini.affa"),
        (r"D:\affan_ai_engine_2_mini\dist\LICENSE.txt", f"{root}/LICENSE.txt"),
        (r"D:\affan_ai_engine_2_mini\dist\README.md", f"{root}/README.md"),
        (r"D:\affan_ai_engine_2_mini\engine_cpp\affa_engine.h", f"{root}/engine_sdk/affa_engine.h"),
        (r"D:\affan_ai_engine_2_mini\build\affa_cli.exe", f"{root}/engine_sdk/affa_cli.exe"),
    ]
    print("Compressing release package with maximum Deflate level 9...")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for src, arc in files:
            print(f"  + Adding {arc} ({os.path.getsize(src) / 1048576:.2f} MB)")
            zf.write(src, arc)

    sz = os.path.getsize(out_zip) / 1048576.0
    print(f"DONE! Created {out_zip} ({sz:.2f} MB)")

if __name__ == "__main__":
    build_release_zip()
