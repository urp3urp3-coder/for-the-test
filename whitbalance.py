# run_preprocess.py
# 사용 예:
# python run_preprocess.py -i ./dataset -o ./whitebalanced --exts .jpg .jpeg .png --workers 8 --suffix _wbclahe --fix-orientation

import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2
import numpy as np

try:
    from PIL import Image, ImageOps
    PIL_OK = True
except Exception:
    PIL_OK = False

# ===== 주어진 전처리 함수들 =====
def gray_world_wb(img_bgr):
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(0) + 1e-6
    gray = means.mean()
    gains = gray / means
    img *= gains
    return np.clip(img, 0, 255).astype(np.uint8)

def lab_clahe_L_only(img_bgr, clip=2.0, tiles=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def preprocess(img_bgr):
    x = gray_world_wb(img_bgr)
    x = lab_clahe_L_only(x)
    return x
# =================================

IMG_EXTS_DEFAULT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(root: Path, allow_exts):
    allow = set(e.lower() for e in allow_exts) if allow_exts else IMG_EXTS_DEFAULT
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in allow:
            yield p

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def load_image_bgr(path: Path, fix_orientation: bool) -> np.ndarray:
    # EXIF 회전 보정이 필요하고 PIL 사용 가능하면 PIL로 읽은 뒤 BGR 변환
    if fix_orientation and PIL_OK:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # EXIF Orientation 적용
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        if img.mode == "RGBA":
            img = img.convert("RGB")
        if img.mode == "L":
            # grayscale -> BGR
            arr = np.array(img)
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            # RGB -> BGR
            arr = np.array(img)
            bgr = arr[..., ::-1].copy()
        return bgr
    else:
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            return None
        # 4채널/1채널 대응
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        return bgr

def save_image(path: Path, img_bgr: np.ndarray, jpg_quality: int):
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    elif path.suffix.lower() == ".png":
        return cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        return cv2.imwrite(str(path), img_bgr)

def process_one(in_path: Path, in_root: Path, out_root: Path, suffix: str, jpg_quality: int, keep_ext: bool, fix_orientation: bool, clip: float, tiles: int):
    # 출력 경로(폴더 구조 보존 + 접미사)
    rel = in_path.relative_to(in_root)
    if keep_ext:
        out_path = (out_root / rel).with_stem(rel.stem + suffix)
    else:
        out_path = (out_root / rel).with_suffix(".jpg")
        out_path = out_path.with_stem(out_path.stem + suffix)

    # 로드
    bgr = load_image_bgr(in_path, fix_orientation)
    if bgr is None:
        return (in_path, "read_error")

    # 전처리 (파라미터 미세 조절 가능)
    def preprocess_param(img):
        x = gray_world_wb(img)
        # tiles는 정사각 그리드만 받게 간단화
        x = lab_clahe_L_only(x, clip=clip, tiles=(tiles, tiles))
        return x

    out = preprocess_param(bgr)

    ensure_parent(out_path)
    ok = save_image(out_path, out, jpg_quality)
    return (in_path, "ok" if ok else "write_error")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="원본 루트 폴더")
    ap.add_argument("-o", "--output", required=True, help="결과 저장 루트(예: ./whitebalanced)")
    ap.add_argument("--exts", nargs="*", default=None, help="처리할 확장자 (예: .jpg .jpeg .png)")
    ap.add_argument("--workers", type=int, default=8, help="동시 처리 개수")
    ap.add_argument("--suffix", type=str, default="_wbclahe", help="파일명 뒤 접미사")
    ap.add_argument("--jpg_quality", type=int, default=95, help="JPG 저장 품질(1-100)")
    ap.add_argument("--keep-ext", action="store_true", help="원본 확장자 유지 (없으면 .jpg로 저장)")
    ap.add_argument("--fix-orientation", action="store_true", help="EXIF 회전 보정 사용(Pillow 필요)")
    ap.add_argument("--clip", type=float, default=2.0, help="CLAHE clipLimit")
    ap.add_argument("--tiles", type=int, default=8, help="CLAHE tileGridSize (정사각, 예: 8 -> (8,8))")
    args = ap.parse_args()

    in_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 이미지 목록 수집
    paths = list(list_images(in_root, args.exts))
    if not paths:
        print("처리할 이미지가 없습니다.")
        return

    # 실행
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_one, p, in_root, out_root,
                args.suffix, args.jpg_quality, args.keep_ext,
                args.fix_orientation, args.clip, args.tiles
            ): p for p in paths
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="processing"):
            results.append(fut.result())

    total = len(results)
    ok = sum(1 for _, s in results if s == "ok")
    print(f"\n총 {total}개 중 성공 {ok}개")
    bad = [(p, s) for p, s in results if s != "ok"]
    if bad:
        print("문제 파일(최대 10개):")
        for p, s in bad[:10]:
            print(f"- {p} : {s}")

if __name__ == "__main__":
    main()
