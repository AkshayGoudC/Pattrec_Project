import cv2

class FaceDetection:
    _model = [
        {
            "_label": "Face",
            "_color": (255, 0, 0),
            "_train_file": r"C:\Semester-4\patt_Rec\Pattrec_Project\dataset\haarcascade_frontalface_default.xml",
            "min_neighbors": 10,
            "_classifier": None,

        },
        {
            "_label": "Eye",
            "_color": (0, 0, 255),
            "_train_file": r"C:\Semester-4\patt_Rec\Pattrec_Project\dataset\haarcascade_eye.xml",
            "min_neighbors": 12,
            "_classifier": None,
        },
        {
            "_label": "Nose",
            "_color": (0, 0, 255),
            "_train_file": r"C:\Semester-4\patt_Rec\Pattrec_Project\dataset\Nariz.xml",
            "min_neighbors": 4,
            "_classifier": None,
        },
        {
            "_label": "Mouth",
            "_color": (255, 255, 255),
            "_train_file": r"C:\Semester-4\patt_Rec\Pattrec_Project\dataset\Mouth.xml",
            "min_neighbors": 20,
            "_classifier": None,
        },
    ]

    _total_count = 0
    _total_face_detection = 0

    def __init__(self):
        self._videoCapture = cv2.VideoCapture(0)

    def _get_img(self):
        _return, _img = self._videoCapture.read()
        return _img

    def release(self):
        _return, _img = self._videoCapture.release()
        cv2.destroyAllWindows()

    def _classify(self):
        _classifier = cv2.CascadeClassifier
        for _temp in self._model:
            _temp["_classifier"] = _classifier(_temp["_train_file"])

    def _draw_boundary(self, img, _item):
        _scale_factor = 1.1
        _features = _item["_classifier"].detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), _scale_factor,
                                                          _item["min_neighbors"], )


        _coords = []


        for _feature in _features:
            _x, _y, _width, _height = _feature
            cv2.rectangle(
                img, (_x, _y), (_x + _width, _y + _height), _item["_color"], 2
            )
            cv2.putText(
                img,
                _item["_label"],
                (_x, _y - 4),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                _item["_color"],
                1,
                cv2.FILLED,
            )
            _coords = [_x, _y, _width, _height]

            return _coords

    def detect(self):
        self._classify()
        _img = self._get_img()


        _coords = []
        _roi_img = None


        _coords = self._draw_boundary(_img, self._model[0])



        if len(_coords) == 4:

            self._total_face_detection += 1

            _roi_img = _img[
                       _coords[1]: _coords[1] + _coords[3],
                       _coords[0]: _coords[0] + _coords[2],
                       ]
            for i, _temp in enumerate(self._model):
                if i > 0:
                    self._draw_boundary(_roi_img, _temp)

        self._total_count += 1
        return _img


    def get_accuracy(self):
        return (self._total_face_detection / self._total_count) * 100


_face_detection = FaceDetection()
while True:
    _img = _face_detection.detect()
    cv2.imshow("face detection", _img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

_face_detection.release()

print("Accuracy", _face_detection._get_img())
