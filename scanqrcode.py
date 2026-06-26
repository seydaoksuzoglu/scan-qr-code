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

# Çözmeden QR Köşelerini Bul
def find_qr_corners(locator, frame):
    """Locate a QR in the frame WITHOUT decoding it, using finder patterns.
    
    Returns a (4, 2) float32 array of corners, or None if no QR is found.
    This is what lets us correct perspective on a code we cannot yet decode.
    """
    ok, points = locator.detect(frame) # locator.detect() QR'ı çözmeye çalışmaz, sadece üç köşedeki finder pattern'lerden (büyük kare işaretler)
    if not ok or points is None:       # konumunu çıkarır. Perspektif bozuk olsa bile çoğu zaman konumu bulabilir — çözmek ayrı, bulmak ayrı.
        return None                    # reshape(4, 2) ile OpenCV'nin (1,4,2) çıktısını sade 4 köşeye indirgeriz.
    return points.reshape(4, 2).astype(np.float32)

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
def decode_image(detector, image):
    """Decode QR test from an image, WeChat first the pyzbar. Text only."""
    # Düzeltilmiş karede artık konuma ihtiyacımız yok (kırmızı dikdörtgeni orijinal köşelerle çizeceğiz),
    # o yüzden bu yardımcı sadece metni döndürüyor. Aynı "WeChat -> pyzbar" sırasını koruyor, mantığı tek yerde toplayıp tekrardan kaçındık.
    texts = [t for t in detector.detectAndDecode(image)[0] if t]
    if texts:
        return texts
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for symbol in zbar_decode(gray, symbols=[ZBarSymbol.QRCODE]):
        texts.append(symbol.data.decode("utf-8", errors="replace"))
    return texts

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
    
    # Stage 2 - perspective correction (runs ONLY when direct solve failed).
    corners = find_qr_corners(locator, frame)
    if corners is None:
        return results # QR not even located -> nothing to do
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
    locator = cv2.QRCodeDetector() # Çözmeden konum bulan detektör

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
    

