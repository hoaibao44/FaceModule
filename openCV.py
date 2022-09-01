import os
from threading import Thread
from time import sleep
import cv2
import matplotlib.pyplot as plt
import numpy
import mediapipe as mp
import face_recognition
from datetime import datetime
import pickle


IMG_SCALE = 4


def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)


class OpenCV():

    def __init__(self, img_path=None) -> None:
        self._handTracker = handTracker()
        if img_path:
            self._img_path = img_path or 'testing.jpg'
            self._img_obj = self.reading_img()
            self._h, self._w = self._img_obj.shape[:2]
            self._click_event_array = []

    def reading_img(self):
        # Reading the image using imread() function
        image = cv2.imread(self._img_path)

        # Extracting the height and width of an image
        h, w = image.shape[:2]

        # Displaying the height and width
        print("Height = {},  Width = {}".format(h, w))
        return image

    def extract_rgb(self):
        image = self._img_obj
        # Extracting RGB values.
        # Here we have randomly chosen a pixel
        # by passing in 100, 100 for height and width.
        (B, G, R) = image[100, 100]

        # Displaying the pixel values
        print("R = {}, G = {}, B = {}".format(R, G, B))

        # We can also pass the channel to extract
        # the value for a specific channel
        B = image[100, 100, 0]
        print("B = {}".format(B))

    def extract_roi(self):
        image = self._img_obj
        # We will calculate the region of interest
        # by slicing the pixels of the image
        roi = image[100: 500, 200: 700]

    def resize_img_h800(self):

        # Calculating the ratio
        ratio = self._img_obj.shape[0] / self._img_obj.shape[1]

        # Creating a tuple containing width and height
        dim = (800, int(800 * ratio))
        print(dim)
        # Resizing the image
        resize_aspect = cv2.resize(self._img_obj, dim)

        return resize_aspect

    def rotating_img(self, img: numpy.ndarray):
        target_img = img

        _h, _w = target_img.shape[:2]

        # Calculating the center of the image
        center = (_w // 2, _h // 2)

        # Generating a rotation matrix
        # the rotate matrix take 3 arg: center point, rotate angle, scale
        matrix = cv2.getRotationMatrix2D(center, -45, 1.0)

        # Performing the affine transformation
        rotated = cv2.warpAffine(target_img, matrix, (_w, _h))

        return rotated

    def show_img(self):
        cv2.imshow('image', self._img_obj)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()

    # function to display the coordinates of
    # of the points clicked on the image
    def _click_event(self, event, x, y, flags, params):
        img = self._img_obj

        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_event_array.append((x, y))
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image', img)

            if len(self._click_event_array) == 1:
                pass
            else:
                # draw a rectangle around the region of interest
                cv2.rectangle(
                    img, self._click_event_array[0], self._click_event_array[1], (0, 255, 0), 2)

                cv2.imshow("image", img)
                self._click_event_array = []

        # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:

            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            cv2.putText(img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x, y), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image', img)

    def interact_img(self):

        cv2.imshow('image', self._img_obj)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', self._click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()

    def detect_face(self, img: numpy.ndarray):
        image = img
        faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags=cv2.CV_HAAR_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return image

    def detect_hand(self, img: numpy.ndarray):
        image = img

        return self._handTracker.handsFinder(image=image)


class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.2, modelComplexity=1, trackCon=0.2):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands  # type: ignore
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # type: ignore

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        handNo = 0
        lmlist = []

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(
                        image, handLms, self.mpHands.HAND_CONNECTIONS)

            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(image, lmlist[8][1:],
                           10, (255, 0, 255), cv2.FILLED)
                cv2.circle(image, lmlist[4][1:],
                           10, (255, 0, 0), cv2.FILLED)

        return image


class FaceRecognite():
    def __init__(self) -> None:
        self.process_this_frame = True
        self.fr_step_value = 2
        self.fr_step = 0
        self._known_face_encoding_file = 'train\\known_encoding.pickle'
        self._known_facename_file = 'train\\known_name.pickle'

        self.known_face_encodings = []
        self.known_face_names = []

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []

        self.initialize()

    def initialize(self):
        if os.path.exists(os.path.join(os.getcwd(), self._known_face_encoding_file)) and os.path.exists(os.path.join(os.getcwd(), self._known_facename_file)):
            print('load from file')
            # load face encoding
            with open(os.path.join(os.getcwd(), self._known_face_encoding_file), 'rb') as f:
                self.known_face_encodings = pickle.load(f)
            f.close()

            # load face label
            with open(os.path.join(os.getcwd(), self._known_facename_file), 'rb') as f:
                self.known_face_names = pickle.load(f)
            f.close()
        else:
            print('create new array')
            # Create arrays of known face encodings and their names
            self.known_face_encodings, self.known_face_names = self.read_input()

        print(self.known_face_names)

    def read_input(self):
        # Load a sample picture and learn how to recognize it.
        bao_image = face_recognition.load_image_file(
            os.path.join(r'D:\Work\OpenCV\DjAngCamBE\videowcv2\train\DT0083\Bao.jpg'))
        bao_face_encoding = face_recognition.face_encodings(bao_image)[0]

        # Load a sample picture and learn how to recognize it.
        vy_image = face_recognition.load_image_file(
            os.path.join(r'D:\Work\OpenCV\DjAngCamBE\videowcv2\train\DT0369\DT0369.jpg'))
        vy_face_encoding = face_recognition.face_encodings(vy_image)[0]

        encoding_array = [bao_face_encoding, vy_face_encoding]
        label_array = ['DT0083', 'DT0369']

        with open(os.path.join(os.getcwd(), self._known_face_encoding_file), 'wb') as f:
            pickle.dump(encoding_array, f)
        f.close()

        with open(os.path.join(os.getcwd(), self._known_facename_file), 'wb') as f:
            pickle.dump(label_array, f)
        f.close()

        return encoding_array, label_array

    def predict_output_as_fr(self, frame):

        known_face_encodings = self.known_face_encodings
        known_face_names = self.known_face_names

        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Resize frame of video to 1/IMG_SCALE size for faster face recognition processing
            small_frame = cv2.resize(
                frame, (0, 0), fx=(1/IMG_SCALE), fy=(1/IMG_SCALE))  # type: ignore

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]  # origin
            # rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(
                rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(
                rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = numpy.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        # print(self.face_names)

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= IMG_SCALE
            right *= IMG_SCALE
            bottom *= IMG_SCALE
            left *= IMG_SCALE

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (255, 0, 0), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)

        return frame


class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):

        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time if elapsed_time > 0 else 0


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False
        self._detect_face = True
        self._myFace = FaceRecognite()
        self._myCV = OpenCV()

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            # self.frame = self._myCV.detect_hand(self.frame)
            if self._detect_face:
                self.frame = self._myFace.predict_output_as_fr(self.frame)

            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True


def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


def noThreading(source=0):
    """Grab and show video frames without multithreading."""

    cap = cv2.VideoCapture(source)
    cps = CountsPerSec().start()

    while True:
        grabbed, frame = cap.read()
        if not grabbed or cv2.waitKey(1) == ord("q"):
            break

        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        cps.increment()


def threadVideoGet(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """

    video_getter = VideoGet(source).start()
    cps = CountsPerSec().start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        cps.increment()


def threadVideoShow(source=0):
    """
    Dedicated thread for showing video frames with VideoShow object.
    Main thread grabs video frames.
    """

    cap = cv2.VideoCapture(source)
    (grabbed, frame) = cap.read()
    video_shower = VideoShow(frame).start()
    cps = CountsPerSec().start()

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed or video_shower.stopped:
            video_shower.stop()
            break

        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame
        cps.increment()


def threadBoth(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(source).start()
    video_shower = VideoShow(video_getter.frame).start()
    video_shower._detect_face = True

    cps = CountsPerSec().start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame

        frame = cv2.flip(frame, 1)
        # frame = myFace.predict_output_as_fr(frame)
        # frame = myCV.detect_hand(frame)

        frame = putIterationsPerSec(frame, cps.countsPerSec())

        video_shower.frame = frame
        cps.increment()


if __name__ == '__main__':

    # cap = cv2.VideoCapture(0)
    # # fr_file_path = 'demo.jpg'
    # myCV = OpenCV()
    # myFace = FaceRecognite()
    # while True:
    #     ret, frame = cap.read()

    #     if not ret:
    #         print("Error: failed to capture image")
    #         break

    #     frame = cv2.flip(frame, 1)

    #     # frame = myFace.predict_output_as_fr(frame)

    #     frame = myCV.detect_hand(frame)
    #     frame = myFace.predict_output_as_fr(frame)

    #     cv2.imshow('Video', frame)

    #     # Hit 'q' on the keyboard to quit!
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         cap.release()
    #         cv2.destroyAllWindows()
    #         break

    print('hello')
    threadBoth()
