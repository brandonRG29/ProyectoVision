import cv2
import train, detect, config, imutils, argparse

def RecognizeFace(image, faceCascade, eyeCascade, faceSize, threshold):
    found_faces = []
    recognizer = train.trainRecognizer("train", faceSize, showFaces=True)
    gray, faces = detect.detectFaces(image, faceCascade, eyeCascade, returnGray=1)
    for ((x, y, w, h), eyedim)  in faces:
        label, confidence = recognizer.predict(cv2.resize(detect.levelFace(gray, ((x, y, w, h), eyedim)), faceSize))
        if confidence < threshold:
            found_faces.append((label, confidence, (x, y, w, h)))

    return found_faces

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagen", required = True,
    help = "Ruta de la imagen para reconocer")
    args = vars(ap.parse_args())

    faceCascade = cv2.CascadeClassifier('cascades/face.xml')
    eyeCascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    faceSize = config.DEFAULT_FACE_SIZE
    threshold = 100
    recognizer = train.trainRecognizer('train', faceSize, showFaces=True)

    cv2.namedWindow("Proyecto", 1)
    capture = cv2.imread(args["imagen"])
    #capture = cv2.VideoCapture(0)
  

  

    while True:
        img = imutils.resize(capture, height=500)
        for (label, confidence, (x, y, w, h)) in RecognizeFace(img, faceCascade, eyeCascade, faceSize, threshold):
            archivo = str(label) + ".txt"
            print("El nombre del archivo es: ",archivo)
            f = open(archivo,'r')
            texto = f.read()
            print(texto)
            f.close()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(img, "{}".format(recognizer.getLabelInfo(label)), (x, y-5), font, 1, (255, 255 ,255), 1, cv2.LINE_AA)
            cv2.putText(img, texto, (x-90, y-50), font, 1, (255, 255 ,255), 1, cv2.LINE_AA)

        print("Se detecto a: %s" % (recognizer.getLabelInfo(label)))
        cv2.imshow("Proyecto", img)
        ch = cv2.waitKey(0)
        if ch == 27:
            break
    cv.destroyAllWindows()
