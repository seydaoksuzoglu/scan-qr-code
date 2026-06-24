"""
İHA İçin QR Kod Çözücü
Mantık: Webcam'den kareyi al -> WeChat ile çöz -> Bulamazsa pyzbar'a düş -> Bulunan QR'ı kırmızı dikdörtgenle çiz, metni yaz.
"""

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

# Dedektörü Kurma
# Neden bunu yapıyoruz? Çünkü detektör nesnesini bir kez kurup tekrar tekrar kullanacağız. Her karede yeniden kurmayacağız.
def build_detector():
    """Create the WeChat QR detector from the local model files."""
    return cv2.wechat_qrcode_WeChatQRCode(
        DETECT_PROTOTXT, DETECT_MODEL, SR_PROTOTXT, SR_MODEL
    )

# Karenin Çözülmesi
# Mantık: WeChat -> pyzbar
def decode_frame(detector, frame):
    """Return a list of (text, points) for QR codes found in the frame.

    Tries WeChat first; if it finds nothing, falls back to pyzbar.
    'points' is a (4, 2) float array of the corner coordinates, or None
    when the decoder does not report a location.
    """
    results = []

    texts, points_list = detector.detectAndDecode(frame) # İki şey döndürür. texts ve her birinin 4 köşesi (points_list)
    for text, pts in zip(texts, points_list):
        if text: # Eğer text dönerse
            results.append((text, np.asarray(pts, dtype=np.float32)))
    
    if results: # Eğer result varsa pyzbar hiç çalıştırılmaz.
        return results
    
    # Fallback: pyzbar works on the grayscale image.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for symbol in zbar_decode(gray, symbols=[ZBarSymbol.QRCODE]):
        text = symbol.data.decode("utf-8", errors="replace")
        pts = np.array([[p.x, p.y] for p in symbol.polygon], dtype=np.float32)
        results.append((text, pts)) # WeChat ile aynı formatta sonuç döner
    
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

            for text, points in decode_frame(detector, frame):
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
    

