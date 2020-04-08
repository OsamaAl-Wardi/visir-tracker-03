import cv2


class CFace():
    id = ""
    text = ""
    area = ""

    def __init__(self, area):
        self.id = -1
        self.text = "Unkown"
        self.area = area

    def getId(self):
        return self.id

    def getText(self):
        return self.text

    def getArea(self):
        return self.area

    def setId(self, newId):
        self.id = newId

    def setArea(self, newArea):
        self.area = newArea

    def setText(self, newText):
        self.text = newText
