import pickle
import random
import socket
import threading
import time
import os
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import torch
from PIL import Image
from skimage.filters import threshold_local
from torchvision import transforms
from build_vocab import Vocabulary
from build_vocab import main
from model import EncoderCNN, DecoderRNN
from transform import four_point_transform

# TODO: FOR APP - add a stop everything button
# TODO: FOR APP - if app can't find server, throw an exception (instead of shutting down)
global msg
device = torch.device('cpu')
counter = 0
global sendtoServer
sendtoServer = ''
global broken
broken = False
global same
same = False
# TODO: Integrate the stop everything button
# TODO: Make sure all items in temp/images get deleted
# TODO: Make sure document_ocr and image_captioning only take 1 picture (instead of 3)
def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def play_loop():
    time.sleep(0.5)
    global msg
    global broken
    global sendtoServer
    global same
    if msg == 'F':
        broken = False
        same = True
        face_recog()
    elif msg == 'M':
        broken = False
        same = True
        money_recog()
    elif msg == 'D' and broken == False and same == False:
        print(broken)
        broken = False
        same = True
        taken = False

        doc_path = get_frame()

        print('b4')
        time.sleep(1.2)
        print('after')
        document_ocr(doc_path)

        time.sleep(0.2)
    elif msg == 'I' and same == False:
        same = True
        broken = False
        image_path = get_frame()
        time.sleep(0.8)
        image_caption(image_path)
    # image_caption('dadinrags.jpg')


def scanner():
    global msg
    global broken

    if msg == 'D':
        print('please pick another mode')
        play_loop()
    elif msg == 'I' or msg == 'M' or msg == 'F':
        play_loop()


def image_caption(img):
    global msg
    global sendtoServer
    connector = False
    if msg != 'I':
        connector = True
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )

    with open('C:\\Users\\Parth\Music\\Assests\\caption_vocab\\caption_vocab\\vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    encoder = EncoderCNN(
        256
    ).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(256, 512, len(vocab), 1)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(
        torch.load('C:\\Users\\Parth\Music\\Assests\\caption_model\\sencoder-5-3000.pkl')
    )
    decoder.load_state_dict(
        torch.load('C:\\Users\\Parth\Music\\Assests\\caption_models\\decoder-5-3000.pkl')
    )

    image = load_image(img, transform)
    image_tensor = image.to(device)

    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    print(sentence)
    sendtoServer = sentence
    image = Image.open(img)
    #    plt.imshow(np.asarray(image))
    if msg == 'I':
        time.sleep(0.01)
    else:
        #  os.remove(img)
        os.remove(image)
        play_loop()


def get_frame():
    count = random.randint(1, 10000)
    global once
    once = True
    time.sleep(0.1)
    print('oncebefore' + str(once))
    if once == True:
        once = False
        print('once after' + str(once))
        time.sleep(0.1)
        once = False
        print("3")
        once = False
        time.sleep(1)
        once = False
        print("2")
        once = False
        time.sleep(1)
        print("1")
        time.sleep(1)
        vidObj = cv2.VideoCapture('http://192.168.0.4:8080')
        ret, img = vidObj.read()
        print('captured picture')
        time.sleep(0.1)
        newpath = "C:\\Users\\Parth\\Music\\Assests\\temp_images\\frame%d.jpg" % count
        cv2.imwrite(newpath, img)
        time.sleep(0.1)
        print("picture saved")

        return newpath


def money_recog():
    global sendtoServer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('C:\\Users\\Parth\\Music\\Assests\\face_model\\trainer.yaml')
    cascadePath = 'C:\\Users\\Parth\Music\\Assests\\face_cascade\\haarcascade_frontalface_alt.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    id = ''

    names = [
        '5 dollars',
        '20 dollars',
        '1 dollar',
        '10 dollars',
        '5 dollars',
        'nope',
    ]

    cam = cv2.VideoCapture('http://192.168.0.4:8080')
    cam.set(3, 640)
    cam.set(4, 480)

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    while True:
        (ret, img) = cam.read()
        # img = cv2.flip(img, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        global msg
        connector = False
        if msg != 'M':
            connector = True
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(
                img, (x, y), (x + w, y + h), (0, 0xFF, 0), 2
            )
            (id, confidence) = recognizer.predict(
                gray[y: y + h, x: x + w]
            )

            if confidence < 100 and confidence > 22:
                id = names[id]
                confidence = '  {0}%'.format(
                    round(100 - confidence)
                )
            else:
                id = ''
                confidence = '  {0}%'.format(
                    round(100 - confidence)
                )
            cv2.putText(
                img,
                str(id),
                (x + 5, y - 5),
                font,
                1,
                (0xFF, 0xFF, 0xFF),
                2,
            )
            cv2.putText(
                img,
                str(confidence),
                (x + 5, y + h - 5),
                font,
                1,
                (0xFF, 0xFF, 0),
                1,
            )
            # if id != '':
            print(id)
            time.sleep(0.3)
            sendtoServer = id
            break
        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xFF
        if connector == True:
            cv2.destroyAllWindows()
            play_loop()
            break


def face_recog():
    global sendtoServer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('C:\\Users\\Parth\\Music\\Assests\\face_model\\trainer.yaml')
    cascadePath = 'C:\\Users\\Parth\Music\\Assests\\face_cascade\\haarcascade_frontalface_alt.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    id = ''

    names = ['Parth', '', '', '', '', '']

    cam = cv2.VideoCapture('http://192.168.0.4:8080')
    cam.set(3, 640)
    cam.set(4, 480)

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    while True:
        (ret, img) = cam.read()
        # img = cv2.flip(img, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        global msg
        connector = False
        if msg != 'F':
            connector = True
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(
                img, (x, y), (x + w, y + h), (0, 0xFF, 0), 2
            )
            (id, confidence) = recognizer.predict(
                gray[y: y + h, x: x + w]
            )

            if confidence < 100 and confidence > 42:
                id = names[id]
                confidence = '  {0}%'.format(
                    round(100 - confidence)
                )
            else:
                id = 'Stranger'
                confidence = '  {0}%'.format(
                    round(100 - confidence)
                )
            cv2.putText(
                img,
                str(id),
                (x + 5, y - 5),
                font,
                1,
                (0xFF, 0xFF, 0xFF),
                2,
            )
            cv2.putText(
                img,
                str(confidence),
                (x + 5, y + h - 5),
                font,
                1,
                (0xFF, 0xFF, 0),
                1,
            )
            # if id != '':
            print(id)
            time.sleep(0.3)
            sendtoServer = id
            # break
        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xFF
        if connector == True:
            cv2.destroyAllWindows()
            play_loop()
            break


def document_ocr(img):
    global sendtoServer
    # TODO: make exception if code can't find all 4 corners
    time.sleep(3)
    global broken
    broken = False
    parth = 0
    count = random.randint(1, 10000)
    image = cv2.imread(img)

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[
           :5
           ]

    broken = False
    screenCnt = 0

    for c in cnts:
        global msg
        global panther
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            print('4 corners found')

    corners = False
    global panther
    # print(type(screenCnt))
    if type(screenCnt) == int:
        corners = True

    if corners == False:
        panther = len(screenCnt)

    if corners == True:
        panther = 2

    if panther != 4 and msg != 'D':
        print('4 corners not found, now exiting')
        cv2.destroyAllWindows()
        play_loop()
        return
    elif panther != 4 and msg == 'D':
        broken = True
        time.sleep(0.5)
        cv2.destroyAllWindows()
        scanner()
        return

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow('Outline', image)

    warped = four_point_transform(
        orig, screenCnt.reshape(4, 2) * ratio
    )

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(
        warped, 11, offset=10, method='gaussian'
    )
    warped = (warped > T).astype('uint8') * 255

    cv2.imshow(
        'Scanned', imutils.resize(warped, height=650)
    )

    cv2.imwrite('C:\\Users\\Parth\Music\\Assests\\temp_ocr\\OCR' + str(count) + '.jpg', warped)

    imPath = 'C:\\Users\\Parth\Music\\Assests\\temp_ocr\\OCR' + str(count) + '.jpg'

    config = '-l eng --oem 1 --psm 3'

    im = cv2.imread(imPath, cv2.IMREAD_COLOR)

    text = pytesseract.image_to_string(im, config=config)
    if text == '':
        print('no text found')
        # print(image)
        # print(img)
        os.remove(imPath)
        os.remove(img)

        cv2.destroyAllWindows()
    else:
        print(text)
    sendtoServer = text
    cv2.waitKey(0)
    if msg == 'D':
        time.sleep(0.01)
    else:
        cv2.destroyAllWindows()
        play_loop()


class ClientThread(threading.Thread):
    def __init__(self, clientAddress, clientsocket):

        threading.Thread.__init__(self)
        self.csocket = clientsocket

        # print("New connection added: ", clientAddress)

    def run(self):
        # print("Connection from : ", clientAddress)
        global sendtoServer
        global msg
        msg = ''
        while True:
            data = self.csocket.recv(2048)
            msg = data.decode()
            park = ''
            if msg == 'bye':
                break
            if (
                    msg != ''
                    and msg != 'ocument OCR\n'
                    and msg != 'mage Captioning\n'
                    and msg != 'oney Recognition\n'
                    and msg != 'acial Recognition\n'
            ):

                print(msg)
                # if sendtoServer != '':
                #  self.csocket.send(bytes(sendtoServer, 'UTF-8'))

                park = 'lezzgo'
            time.sleep(1)
            play_loop()
        print(
            "Client at ", clientAddress, " disconnected..."
        )


LOCALHOST = "192.168.0.14"
PORT = 8080
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((LOCALHOST, PORT))
print("Server started")
print("Waiting for client request..")
while True:
    server.listen(1)
    clientsock, clientAddress = server.accept()
    newthread = ClientThread(clientAddress, clientsock)
    newthread.start()
