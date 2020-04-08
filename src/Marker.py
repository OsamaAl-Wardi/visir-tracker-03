import cv2

class Marker():
    def MarkFaces(image, faces ):
        for face in faces:
            area = face.getArea()
            image = cv2.rectangle(image, (area[0], area[1]), (area[0]+area[2], area[1] + area[3]), (255, 0, 255), 3)
            cv2.putText(image, face.getText() + " ID: " + str(face.getId()), (area[0], area[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 5)


    def MarkPoints(image, points, color):
        for point in points:
            cv2.circle(image, point, 3, color, 2)

  #  def MarkVecOff(image, hflow, vflow):
            #code here

    def markGUI(image):
        #implement GUI
        #show FPS
        for x in image.shape[0]:
            vector = image.row[x]
            for y in image.shape[1]:
                k =  x / image.cols
                vector[x] = cv2.Vec3b(k * 255, 0, 255 - k * 255)
        cv2.circle(image, (image.cols / 2, image.rows /2), 50, (100, 255, 100), 5)
        cv2.GaussianBlur(image, image, (17, 17), 50)
        cv2.putText(image, "HCI", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)


