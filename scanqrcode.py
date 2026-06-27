"""
İHA İçin QR Kod Çözücü
Mantık: Webcam'den kareyi al -> WeChat ile çöz -> Bulamazsa pyzbar'a düş -> Bulunan QR'ı kırmızı dikdörtgenle çiz, metni yaz.
"""
import os
import logging
import warnings

os.environ["YOLO_VERBOSE"] = "False"
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from qrdet import QRDetector
from pathlib import Path

import cv2
import numpy as np 
from pyzbar.pyzbar import decode as zbar_decode, ZBarSymbol

# Model Yolları
MODELS_DIR = Path(__file__).resolve().parent / "models"
DETECT_PROTOTXT = str(MODELS_DIR / "detect_2021nov.prototxt")
DETECT_MODEL = str(MODELS_DIR / "detect_2021nov.caffemodel") 
SR_PROTOTXT = str(MODELS_DIR / "sr_2021nov.prototxt")
SR_MODEL = str(MODELS_DIR / "sr_2021nov.caffemodel")

# Renk ve Kamera Sabitleri
# Competition rule: lock-on quadrangle must be pure red (#FF0000), line <= 3 px.
QUAD_COLOR = (0, 0, 255) # BGR for #FF0000 (OpenCV renkleri BGR sırasıyla tutar.)
QUAD_THICKNESS = 2

CAMERA_INDEX = 0 # Harici kamera takarsak 1 yaparız.

MIN_QR_SIDE = 80 # px; below thi a located QR is too small to decode -> skip homography

# Dedektörü Kurma
# Neden bunu yapıyoruz? Çünkü detektör nesnesini bir kez kurup tekrar tekrar kullanacağız. Her karede yeniden kurmayacağız.
def build_detector():
    """Create the WeChat QR detector from the local model files."""
    return cv2.wechat_qrcode_WeChatQRCode(
        DETECT_PROTOTXT, DETECT_MODEL, SR_PROTOTXT, SR_MODEL
    )
# QR Bulucu (qrdet / YOLOv8) - cv2.QRCodeDetector'ın yerini aldı
def build_locator(model_size="n"):
    """qrdet YOLOv8 QR locator. 'n': nano for (Raspberry Pi)"""
    return QRDetector(model_size=model_size, conf_th=0.5)

# QR'ı Bul (çözmeden) - qrdet/YOLO konum bulur
def find_qr_corners(locator, frame):
    """Locate the confident QR with qrdet (YOLO); return (4, 2) corners or None.

    qrdet, perspektif/bulanıklık altında QR bölgesini cv2.QRCodeDetector'dan çok daha
    iyi bulur, eski dedektör sahte tespit üretiyordu. 
    
    qrdet birden fazla QR dönebilir, en yüksek confidence'lının quad_xy (4 köşe) alanını alıp doğrudan
    warp_qr'a veriyoruz. is_bgr=True çünkü OpenCV için BGR gerekli.
    """
    detections = locator.detect(image=frame, is_bgr=True) # cv2 kareleri BGR
    if not detections:
        return None
    best = max(detections, key=lambda d: d["confidence"])
    return best["quad_xy"].astype(np.float32)

def quad_side(corners):
    """Average edge length (px) of the located quad - a cheap size proxy."""
    sides = [np.linalg.norm(corners[i] - corners[(i+1) % 4]) for i in range(4)]
    return float(np.mean(sides))

# Homografi ile QR'ı Düz Kareye Çevir
def warp_qr(frame, corners, size=320, quiet=24):
    """Warp the 4 QR corners onto a flat front-facing square.

    A white border (quiet zone) is added because QR decoders require
    empty margin around the code to lock on.
    """
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size -1]], 
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, dst) # getPerspectiveTransform 4 eğik köşeyi 4 düz köşeyle eşleyen homografi matrisini hesaplar.
    warped = cv2.warpPerspective(frame, transform, (size, size)) # warpPerspective, bu matrise görüntüyü "tam karşıdan bakıyormuş gibi" düzeltir.
    warped = cv2.copyMakeBorder( # copyMakeBorder ile etrafına beyaz kenar (quiet zone) ekliyoruz. QR standardı kodun çevresinde boş alan ister.
        warped, quiet, quiet, quiet, quiet, # Düzeltilmiş kare ekranı tam doldurursa çözücüler kilitlenemez, bu yüzden beyaz çerçeve şart.
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )
    return warped

# Bir Görüntüyü Çöz (sadece metin)
# Ön İşleme (kontrast iyileştirme)
# Mantık: Düzeltilmiş kareyi önce ham, sonra CLAHE (yerel kontrast iyileştirme) uygulanmış haliyle çözmeyi dene. Düşük kontrast/gölgeli
# QR'larda modüllerin siyah-beyaz ayrımını netleştir. Soluk duruma yardım eder. Parlama çözülmez.
def decode_image(detector, image):
    """Decode QR test from an image. Tries the raw image first, then a contrast-enhanced (CLAHE) version.
    WeChat first pyzbar second, on the each."""
    # Düzeltilmiş karede artık konuma ihtiyacımız yok (kırmızı dikdörtgeni orijinal köşelerle çizeceğiz),
    # o yüzden bu yardımcı sadece metni döndürüyor. Aynı "WeChat -> pyzbar" sırasını koruyor, mantığı tek yerde toplayıp tekrardan kaçındık.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # createCLAHE görüntüyü küçük karelere bölüp kontrastı ayrı ayrı artırır.
                                                                # global kontrasttan farkı, bir köşedeki gölge diğer köşeyi bozmaz. 
                                                                # Eğik yer QR'ında ışık dağılımı eşit olmadığı için bu uygun.
    enhanced = clahe.apply(gray)
    for candidate in (image, enhanced): # İki aday deniyoruz (image, enhanced): biri çözerse hemen dönüyoruz. C eşiğinden dolayı bu büyük 
                                        # QR'lı karelerde çalışır küçük karelerde değil.
        texts = [t for t in detector.detectAndDecode(candidate)[0] if t]
        if texts:
            return texts
        gray_c = candidate if candidate.ndim == 2 else cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY) 
        # candidate.ndim == 2 kontrolü enhanced tek kanal (gri), tekrar griye çevirmeye çalışırsak hata verir; onu atlıyoruz.
        for symbol in zbar_decode(gray_c, symbols=[ZBarSymbol.QRCODE]):
            text = symbol.data.decode("utf-8", errors="replace")
            if text:
                return [text]
    return []
    
    

# Karenin Çözülmesi
# Mantık: WeChat -> pyzbar
def decode_frame(detector, locator, frame):
    """WeChat/pyzbar direct; only on failure, perspective-correct and retry.

    Returns a list of (text, points). Homography runs ONLY when the direct
    decode finds nothing, to protect onboard FPS on the Raspberry Pi.
    """
    results = []

    # Stage 1 - direct decode (gives us locations for free).
    texts, points_list = detector.detectAndDecode(frame) # İki şey döndürür. texts ve her birinin 4 köşesi (points_list)
    for text, pts in zip(texts, points_list):
        if text: # Eğer text dönerse
            results.append((text, np.asarray(pts, dtype=np.float32)))
    
    if not results:
        # Fallback: pyzbar works on the grayscale image.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for symbol in zbar_decode(gray, symbols=[ZBarSymbol.QRCODE]):
            text = symbol.data.decode("utf-8", errors="replace")
            pts = np.array([[p.x, p.y] for p in symbol.polygon], dtype=np.float32)
            results.append((text, pts)) # WeChat ile aynı formatta sonuç döner
    
    if results:
        return results # direct solve worked -> skip homography (performance rule)
    
    # Stage 2 - perspective correction (only when direct solve failed).
    corners = find_qr_corners(locator, frame)
    if corners is None:
        return results # QR not even located -> nothing to do
    if quad_side(corners) < MIN_QR_SIDE:
        return results # located but too small to decode -> skip expensive warp
    warped = warp_qr(frame, corners)
    for text in decode_image(detector, warped):
        results.append((text, corners)) # draw the box with ORIGINAL-frame corners
    return results

# Çizim
def draw_quad(frame, points):
    """Draw the QR boundary as a closed red polygon."""
    if points is None or len(points) < 4:
        return
    poly = points.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [poly], isClosed=True, color=QUAD_COLOR,
                  thickness=QUAD_THICKNESS) # cv2.polylines tam sayı koordinat ister. O yüzden yukardaki dönüşümü yaptık. 
    # isClosed=True, dikdörtgeni kapatır (son köşeden ilk köşeye çizgi). Şimdilik basit ilerde telemetri/hedef alan ile zenginleşecek.

# main Fonksiyonu
def main():
    detector = build_detector()
    locator = build_locator()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {CAMERA_INDEX} could not be opened")
    
    print("Reading... press 'q' to quit.")
    last_text = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            for text, points in decode_frame(detector, locator, frame):
                draw_quad(frame, points)
                if text != last_text:
                    print("QR: ", text)
                    last_text = text

            cv2.imshow("ReadQRCode", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    

