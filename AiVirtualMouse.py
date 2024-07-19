import cv2
import numpy as np
import pyautogui as pag
import time
from HandTrackingModule import handDetector

def main():
    wCam, hCam = 640, 480
    frameR = 150  # Frame Reduction
    smoothening = 7

    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    wScr, hScr = pag.size()

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0
    detector = handDetector(maxHands=1)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        # 1. Find hand landmarks
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        # 2. Get the tip of the index, middle, and thumb fingers
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]  # Index finger tip
            x2, y2 = lmList[12][1:] # Middle finger tip
            x0, y0 = lmList[4][1:]  # Thumb tip

            # 3. Check which fingers are up
            fingers = detector.fingersUp()

            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            # 4. Only Index Finger = Moving Mode
            if fingers[1] == 1 and fingers[2] == 0:
                # 5. Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # 6. Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # 7. Move Mouse
                pag.moveTo(wScr - clocX, clocY)
                plocX, plocY = clocX, clocY
                cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)

            # 8. Both index and middle fingers are up: Right-clicking mode
            if fingers[1] == 1 and fingers[2] == 1:
                # 9. Find distance between index and middle fingers
                length, img, lineInfo = detector.findDistance(8, 12, img)

                # 10. Right-click mouse if distance is short
                if length < 40 :
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    pag.click()

            # 11. Thumb and index finger are up: Left-clicking mode
            if fingers[1] == 1 and fingers[0] == 1 and fingers[2] == 0:

                # 12. Find distance between thumb and index fingers
                length, img, lineInfo = detector.findDistance(4, 8, img)

                # 13. Left-click mouse if distance is short
                if length < 40 :
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    pag.rightClick()

        # 14. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # 15. Display
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord("a"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
