#File run.py chứa các hàm để thực thi mô hình
# File c1.caffemodel: chứa mô hình màu hóa phương pháp 1
# File c2_norebal.caffemodel: chứa mô hình màu hóa phương pháp 2. Phương pháp này thêm tính năng cân bằng màu
# File c1.prototxt chứa định nghĩa các lớp (player) của mạng nowrron của mô hình màu hóa
# File c3.npy là file đóng gói giữ liệu dưới dạng mảng để thực thi mô hình màu hóa npy viết tắt numpy
# là thư viện sử dụng ngon ngữ python, chuyên dùng để xử lý dữ liệu dạng mảng
# lưu ý: ảnh là dữ liệu mảng 2 chiều
# Các file này tham khảo từ bài báo mà đã trình bày trong báo cáo

import cv2    # dyngf để sử dụng opencv: thư viện xử lý ảnh
import sys
import time

import imutils   # dùng để sxwr lý video
import numpy as np   # dùng để xử lý dữ liệu mảng
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog # dùng để xứ lý giao diện PyQT
from imutils.video import FileVideoStream
from imutils.video import FPS

from gui import *

class Window(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        #self.retranslateUi(self)


    def open(self):   # Định nghĩa hàm để mở và hiển thị ảnh lên label
        options = QFileDialog.DontUseNativeDialog
        self.filename, _ = QFileDialog.getOpenFileName(self, "Mở File", "", "Kiểu File (*.jpg)", options=options)
        if self.filename:
            pixmap1 = QtGui.QPixmap(self.filename)

            if pixmap1.width() > self.label_anhgoc.width():
                self.pixmap = pixmap1.scaled(451, 255)
                self.label_anhgoc.setPixmap(self.pixmap)
                self.label_anhgoc.resize(451, 255)
            else:
                self.label_anhgoc.setPixmap(pixmap1)
                #self.label_anhgoc.resize(pixmap1.width(), pixmap1.height())


    def mauhoa_anh1(self):   # Định nghĩa hàm để màu hóa ảnh bằng phương pháp 1 và hiển thị ảnh lênh label
        print("[INFO] Nạp mô hình phương pháp 1...")
        net = cv2.dnn.readNetFromCaffe("c1.prototxt", "c2.caffemodel")
        pts = np.load("c3.npy")

        # add the cluster centers as 1x1 convolutions to the model
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        image = cv2.imread(self.filename)

        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")

        temp = self.filename.find(".")
        temp1 = self.filename[:temp] + "color_pp1" + self.filename[temp:]
        cv2.imwrite(temp1, colorized)  # Lệnh này lưu ảnh màu hóa

        cv2.imwrite(temp1, colorized)   # Lệnh này lưu ảnh màu hóa
        self.label_anhmau.setPixmap(QtGui.QPixmap(temp1))
        pixmap2 = QtGui.QPixmap(temp1)
        if pixmap2.width() > self.label_anhgoc.width():
            self.pixmapnew = pixmap2.scaled(451, 255)
            self.label_anhmau.setPixmap(self.pixmapnew)
            self.label_anhmau.resize(451, 255)
        else:
            self.label_anhmau.setPixmap(pixmap2)
        cv2.waitKey(0)

    def mauhoa_anh2(self):   # Hàm để màu hóa ảnh bằng phương pháp 2 và hiển thị ảnh. PP2 có cân bằng màu
        print("[INFO] Nạp mô hình phương pháp 2...")
        net = cv2.dnn.readNetFromCaffe("c1.prototxt", "c2_norebal.caffemodel")
        pts = np.load("c3.npy")

        # add the cluster centers as 1x1 convolutions to the model
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        image = cv2.imread(self.filename)

        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")
        temp = self.filename.find(".")
        temp1 = self.filename[:temp]+"color_pp2"+self.filename[temp:]
        cv2.imwrite(temp1, colorized)   # Lệnh này lưu ảnh màu hóa
        self.label_anhmau.setPixmap(QtGui.QPixmap(temp1))
        pixmap2 = QtGui.QPixmap(temp1)
        if pixmap2.width() > self.label_anhgoc.width():
            self.pixmapnew = pixmap2.scaled(451, 255)
            self.label_anhmau.setPixmap(self.pixmapnew)
            self.label_anhmau.resize(451, 255)
        else:
            self.label_anhmau.setPixmap(pixmap2)
        cv2.waitKey(0)


    ''' Không dùng đoạn code này
    def open_video(self):
        options = QFileDialog.DontUseNativeDialog
        self.filename2, _ = QFileDialog.getOpenFileName(self, "Mở File", "", "Kiểu File (*.mp4)", options=options)
        print("[INFO] Mở file...")
        vs = cv2.VideoCapture(self.filename2)

        while True:
            ret,frame = vs.read()
            self.frame = imutils.resize(frame,width=500)
            scaled = frame.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            resized = cv2.resize(lab, (224, 224))
            self.QtImg = QtGui.QImage(self.frame, self.frame.shape[1], self.frame.shape[0], QtGui.QImage.Format_RGB888)
            # Display the image to the label;
            self.label_videogoc.resize(QtCore.QSize(self.frame.shape[1], self.frame.shape[0]))
            self.label_videogoc.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.label_videogoc.resize(451, 255)
            key = cv2.waitKey(1) & 0xFF
            # Trong quá trình màu hóa nhấn phím q để thoát
            if key == ord("q"):
                break
        vs.release()
        cv2.destroyAllWindows()
        '''

    def mauhoa_video1(self):  # Định nghĩa hàm màu hóa video bằng phương pháp 1
        options = QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Mở File", "", "Kiểu File (*.mp4)", options=options)
        # initialize the video stream and allow the camera
        # sensor to warmup
        print("[INFO] bắt đầu đọc file và xử lý...")
        fvs = FileVideoStream(filename).start()
        time.sleep(1.0)
        fps = FPS().start()
        # Specify the paths for the 2 model files
        protoFile = "c1.prototxt"
        weightsFile = "c2.caffemodel"
        balweightsFile = "c2_norebal.caffemodel"

        # Load the cluster centers
        pts_in_hull = np.load('c3.npy')

        # Read the network into Memory
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        balnet = cv2.dnn.readNetFromCaffe(protoFile, balweightsFile)
        # populate cluster centers as 1x1 convolution kernel
        pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
        ### TWEAK HYPER-PARAMETERS HERE?
        net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
        net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

        # from opencv2 sample
        W_in = 224
        H_in = 224

        numframe = 24  # Thiết lập số khung hình của video 24 hoặc 30
        tempCodeVideo="MJPG" #MJPG đối với mp4; XVID đối với avi
        CodeVideo = cv2.VideoWriter_fourcc(*tempCodeVideo)
        writer = None
        (h, w) = (None, None)
        zeros = None
        counter = 0

        # loop over frames from the video file stream
        while fvs.more():
            counter += 1
            frame = fvs.read()

            frame = imutils.resize(frame, width=500)
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayframe = np.dstack([grayframe, grayframe, grayframe])

            if writer is None:
                # store the image dimensions, initialzie the video writer,
                # and construct the zeros array
                temp = filename.find(".")
                temp1 = filename[:temp] + "color_pp1" + filename[temp:]
                (h, w) = frame.shape[:2]
                writer = cv2.VideoWriter(temp1, CodeVideo, numframe, (w, h), True)
                zeros = np.zeros((h, w), dtype="uint8")

            img_rgb = (grayframe[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
            img_l = img_lab[:, :, 0]  # pull out L channel

            # resize lightness channel to network input size
            img_l_rs = cv2.resize(img_l, (W_in, H_in))  #
            img_l_rs -= 50  # subtract 50 for mean-centering

            # resize lightness channel to network input size


            net.setInput(cv2.dnn.blobFromImage(img_l_rs))
            ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))  # this is our result

            (H_orig, W_orig) = img_rgb.shape[:2]  # original image size
            ab_dec_us = cv2.resize(ab_dec, (W_orig, H_orig))
            img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)  # concatenate with original image L
            img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)

            output = np.zeros((h , w , 3), dtype="uint8")
            output2 = np.zeros((h , w , 3), dtype="uint8")


            output[0:h, 0:w] = frame
            output2[0:h , 0:w ] = img_bgr_out * 255


            writer.write(output2)

            self.QtImg2 = QtGui.QImage(output, output.shape[1], output.shape[0], QtGui.QImage.Format_BGR888)
            # Display the image to the label;
            self.label_videogoc.resize(QtCore.QSize(output.shape[1], output.shape[0]))
            self.label_videogoc.setPixmap(QtGui.QPixmap.fromImage(self.QtImg2))
            self.label_videogoc.resize(451, 255)

            self.QtImg3 = QtGui.QImage(output2, output2.shape[1], output2.shape[0], QtGui.QImage.Format_BGR888)
            # Display the image to the label;
            self.label_videomau.resize(QtCore.QSize(output2.shape[1], output2.shape[0]))
            self.label_videomau.setPixmap(QtGui.QPixmap.fromImage(self.QtImg3))
            self.label_videomau.resize(451, 255)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            cv2.waitKey(1)
            fps.update()

        fps.stop()
        # do a bit of cleanup
        print("[INFO] cleaning up...")
        cv2.destroyAllWindows()
        fvs.stop()
        writer.release()


    def mauhoa_video2(self):    #màu hóa video bằng PP2
        options = QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Mở File", "", "Kiểu File (*.mp4)", options=options)
        # initialize the video stream and allow the camera
        # sensor to warmup
        print("[INFO] Bắt đầu đọc file và xử lý màu hóa...")
        fvs = FileVideoStream(filename).start()
        time.sleep(1.0)
        fps = FPS().start()

        # Specify the paths for the 2 model files
        protoFile = "c1.prototxt"
        weightsFile = "c2.caffemodel"
        balweightsFile = "c2_norebal.caffemodel"

        # Load the cluster centers
        pts_in_hull = np.load('c3.npy')

        # Read the network into Memory
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        balnet = cv2.dnn.readNetFromCaffe(protoFile, balweightsFile)
        # populate cluster centers as 1x1 convolution kernel
        pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)

        ### TWEAK HYPER-PARAMETERS HERE?


        balnet.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
        balnet.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

        # from opencv2 sample
        W_in = 224
        H_in = 224

        numframe = 30  # Thiết lập số khung hình của video
        tempCodeVideo="XVID" #MJPG đối với mp4; XVID đối với avi
        CodeVideo = cv2.VideoWriter_fourcc(*tempCodeVideo)

        writer = None
        (h, w) = (None, None)
        #zeros = None


        # loop over frames from the video file stream
        while fvs.more():

            frame = fvs.read()
            frame = imutils.resize(frame, width=500)
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayframe = np.dstack([grayframe, grayframe, grayframe])
            if writer is None:
                # store the image dimensions, initialzie the video writer,
                # and construct the zeros array
                temp = filename.find(".")
                temp1 = filename[:temp] + "color_pp2" + filename[temp:]
                (h, w) = frame.shape[:2]
                writer = cv2.VideoWriter(temp1, CodeVideo, numframe, (w , h ),True)
                #zeros = np.zeros((h, w), dtype="uint8")

            img_rgb = (grayframe[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
            img_l = img_lab[:, :, 0]  # pull out L channel

            # resize lightness channel to network input size
            img_l_rs = cv2.resize(img_l, (W_in, H_in))  #
            img_l_rs -= 50  # subtract 50 for mean-centering

            balimg_rgb = (grayframe[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
            balimg_lab = cv2.cvtColor(balimg_rgb, cv2.COLOR_RGB2Lab)
            balimg_l = balimg_lab[:, :, 0]  # pull out L channel

            # resize lightness channel to network input size
            balimg_l_rs = cv2.resize(balimg_l, (W_in, H_in))  #
            balimg_l_rs -= 50  # subtract 50 for mean-centering

            balnet.setInput(cv2.dnn.blobFromImage(balimg_l_rs))
            balab_dec = balnet.forward()[0, :, :, :].transpose((1, 2, 0))  # this is our result

            (H_orig, W_orig) = balimg_rgb.shape[:2]  # original image size
            balab_dec_us = cv2.resize(balab_dec, (W_orig, H_orig))
            balimg_lab_out = np.concatenate((balimg_l[:, :, np.newaxis], balab_dec_us),axis=2)  # concatenate with original image L
            balimg_bgr_out = np.clip(cv2.cvtColor(balimg_lab_out, cv2.COLOR_Lab2BGR), 0, 1)

            output3 = np.zeros((h , w  , 3), dtype="uint8")
            output3[0:h , 0:w] = balimg_bgr_out * 255
            writer.write(output3)

            self.QtImg2 = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_BGR888)
            # Display the image to the label;
            self.label_videogoc.resize(QtCore.QSize(frame.shape[1], frame.shape[0]))
            self.label_videogoc.setPixmap(QtGui.QPixmap.fromImage(self.QtImg2))
            self.label_videogoc.resize(451, 255)

            self.QtImg3 = QtGui.QImage(output3, output3.shape[1], output3.shape[0], QtGui.QImage.Format_BGR888)
            # Display the image to the label;
            self.label_videomau.resize(QtCore.QSize(output3.shape[1], output3.shape[0]))
            self.label_videomau.setPixmap(QtGui.QPixmap.fromImage(self.QtImg3))
            self.label_videomau.resize(451, 255)


            cv2.waitKey(1)
            fps.update()

        fps.stop()
        # do a bit of cleanup
        print("[INFO] Hoàn thành...")
        cv2.destroyAllWindows()
        fvs.stop()
        writer.release()

if __name__ == "__main__":   #Hàm chính của chương trình gọi thực thi giao diện và chương trình
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())