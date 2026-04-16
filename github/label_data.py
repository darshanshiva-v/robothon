import os
import subprocess

img_folder = "dataset"
label_folder = "labels"

os.makedirs(label_folder, exist_ok=True)

images = sorted(os.listdir(img_folder))

for img_name in images:
    img_path = os.path.join(img_folder, img_name)

    print(f"\n👉 Labeling {img_name}")

    # 🔥 Convert WSL path → Windows path
    full_path = os.path.abspath(img_path)
    win_path = full_path.replace("/home/pragathi", r"\\wsl$\Ubuntu\home\pragathi")
    win_path = win_path.replace("/", "\\")

    # Open image in Windows viewer
    subprocess.run(["cmd.exe", "/c", "start", "", win_path])

    # ---- USER INPUT ----
    print("Enter PLUG box (x y w h):")
    px, py, pw, ph = map(float, input().split())

    print("Enter SOCKET box (x y w h):")
    sx, sy, sw, sh = map(float, input().split())

    # Assume image size (update if different)
    w, h = 640, 480

    # Convert to YOLO format
    def convert(x, y, bw, bh):
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        bw = bw / w
        bh = bh / h
        return xc, yc, bw, bh

    px, py, pw, ph = convert(px, py, pw, ph)
    sx, sy, sw, sh = convert(sx, sy, sw, sh)

    label_file = os.path.join(label_folder, img_name.replace(".png", ".txt"))

    with open(label_file, "w") as f:
        f.write(f"0 {px} {py} {pw} {ph}\n")  # plug
        f.write(f"1 {sx} {sy} {sw} {sh}\n")  # socket

    print(f"✅ Saved: {label_file}")
