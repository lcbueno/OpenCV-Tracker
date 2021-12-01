import cv2


tracker = cv2.TrackerKCF_create()


video = cv2.VideoCapture('race.mp4')

ok, frame = video.read()

bbox = cv2.selectROI(frame)

#print(bbox)

ok = tracker.init(frame, bbox)

print(ok)

while True:
    ok, frame = video.read()
    #print(ok)
    if not ok:
        break
    ok, bbox = tracker.update(frame)
    #print(bbox)

    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2, 1)
    else:
        cv2.putText(frame, 'Erro', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Rastreamento', frame)
    if cv2.waitKey(1) & 0XFF == 27:
        break


# this algorithm is not suitable for working with very fast images;
# below is what we should install through the finish if there are problems running Tracker;


# esse algoritimo não é indicado para trabalhar com imagens muito rápidas;
# abaixo é o que devemos instalar atravaés do terminar se houver probelmas na execução do Tracker;

'''(Rastreamento) C:\Users\advlu\pythonProject1>pip3 install opencv-python
Collecting opencv-python
  Downloading opencv_python-4.5.4.60-cp39-cp39-win_amd64.whl (35.1 MB)
     |████████████████████████████████| 35.1 MB 68 kB/s
Collecting numpy>=1.19.3
  Using cached numpy-1.21.4-cp39-cp39-win_amd64.whl (14.0 MB)
Installing collected packages: numpy, opencv-python
Successfully installed numpy-1.21.4 opencv-python-4.5.4.60
(Rastreamento) C:\Users\advlu\pythonProject1> pip install opencv-python
Requirement already satisfied: opencv-python in c:\users\advlu\pythonproject1\rastreamento\lib\site-packages (4.5.4.60)
Requirement already satisfied: numpy>=1.19.3 in c:\users\advlu\pythonproject1\rastreamento\lib\site-packages (from opencv-python) (1.21.4
Requirement already satisfied: numpy>=1.19.3 in c:\users\advlu\pythonproject1\rastreamento\lib\site-packages (from opencv-python) (1.21.4
)

(Rastreamento) C:\Users\advlu\pythonProject1>
(Rastreamento) C:\Users\advlu\pythonProject1>pip install opencv-contrib-python
Collecting opencv-contrib-python
  Downloading opencv_contrib_python-4.5.4.60-cp39-cp39-win_amd64.whl (42.0 MB)
     |████████████████████████████████| 42.0 MB 74 kB/s
Requirement already satisfied: numpy>=1.19.3 in c:\users\advlu\pythonproject1\rastreamento\lib\site-packages (from opencv-contrib-python)
 (1.21.4)
Installing collected packages: opencv-contrib-python
Successfully installed opencv-contrib-python-4.5.4.60

(Rastreamento) C:\Users\advlu\pythonProject1>
 '''