import sys
import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from ssd import build_ssd
from data import BaseTransform
import threading
import time
detections = None
frame = None


class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cuda = False
        if self.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        trained_model_path = 'weights/ssd300_COCO_18000.pth'
        num_classes = 1 + 1  # +1 background
        self.net = build_ssd('test', 300, num_classes)  # initialize SSD
        if not self.cuda:
            self.net.load_state_dict(torch.load(
                trained_model_path,
                map_location=lambda storage,
                loc: storage)
            )
        else:
            self.net.load_state_dict(torch.load(trained_model_path))
        self.net.eval()
        if self.cuda:
            self.net = self.net.cuda()
            cudnn.benchmark = True
        self.transform = BaseTransform(self.net.size, (104, 117, 123))

    def run(self):
        global detections, frame
        while True:
            if frame is None:
                continue
            detections = self.predict(frame)


    def predict(self, frame):
        print('Predicting..')
        img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        x = torch.from_numpy(self.transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if self.cuda:
            x = x.cuda()
        y = self.net(x)
        return y.data


cap = cv2.VideoCapture(1)
if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

Modelthread = MyThread()
Modelthread.start()


vidcap = cv2.VideoCapture('mylove.mp4')
print(vidcap)

while (True):
    ret, frame = vidcap.read()
    print(ret)
    if detections is not None:
        headCount = sum(detections[0][1][:, 0] >= 0.27)
        cv2.putText(
            frame, "Face Count: {}".format(headCount),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        scale = torch.Tensor([frame.shape[1], frame.shape[0],
                             frame.shape[1], frame.shape[0]])

        j = 0
        while detections[0, 1, j, 0] >= 0.27:
            score = detections[0, 1, j, 0]
            pt = (detections[0, 1, j, 1:] * scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])
            XMin, YMin, XMax, YMax = coords
            w = XMax - XMin + 1
            h = YMax - YMin + 1
            cv2.rectangle(frame, (XMin, YMin), (XMax, YMax), (0, 0, 255), 2)
            j += 1
        cv2.imshow('Image', frame)
        time.sleep(10)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
frame = None
Modelthread.join()
cv2.destroyAllWindows()
sys.exit()
