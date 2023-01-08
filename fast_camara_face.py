#conda install git
#pip install git+https://github.com/elliottzheng/face-detection.git@master
import cv2
from face_detection import RetinaFace

def find_face(img,bbox_list,landmark=False):
    img = cv2.rectangle(img, (int(bbox_list[2]), int(bbox_list[3])), (int(bbox_list[0]), int(bbox_list[1])), (255, 255, 255), 2)
            
    return img
def nothing(x):
    pass

def biggest_face(bbox_list):
    bbox_a = 0
    for i in range(len(bbox_list)):
        temp = (bbox_list[i][0][2]-bbox_list[i][0][0])*(bbox_list[i][0][3]-bbox_list[i][0][1])
        if temp > bbox_a:
            bbox = bbox_list[i][0].astype(int)
            bbox_a = temp
    return(bbox)
def cut_face(img,bbox):
    img2 = img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    return(img2)
def bind_face(img,face,bbox):
    img[bbox[1]:bbox[3],bbox[0]:bbox[2]] = face
    return(img)


# 選擇第一隻攝影機
cap = cv2.VideoCapture(0)
#-1代表使用cpu
detector = RetinaFace(gpu_id=-1)

winName='beauty camera'
cv2.namedWindow(winName)
cv2.createTrackbar('sigmaColor',winName,0,100,nothing)
cv2.setTrackbarPos('sigmaColor',winName, 10)
cv2.createTrackbar('sigmaSpace',winName,0,15,nothing)
cv2.setTrackbarPos('sigmaSpace',winName, 5)
#cv2.createTrackbar('sharpe',winName,3,7,nothing)
cv2.createTrackbar('lightness',winName,-100,200,nothing)
cv2.setTrackbarPos('lightness',winName, 0)
cv2.createTrackbar('saturation',winName,0,300,nothing)

while(True):
  # 從攝影機擷取一張影像
  ret, frame = cap.read()
  sigmaColor=cv2.getTrackbarPos('sigmaColor',winName)
  sigmaSpace=cv2.getTrackbarPos('sigmaSpace',winName)
  #sharpe=cv2.getTrackbarPos('sharpe',winName)
  sharpe=3
  lightness=cv2.getTrackbarPos('lightness',winName)
  saturation=cv2.getTrackbarPos('saturation',winName)



  faceb = detector(frame)
  faceb = biggest_face(faceb)

  hlsImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
  temp = (1 + lightness / 200.0) * hlsImg[:, :, 1]
  temp[temp > 255] = 255  
  hlsImg[:, :, 1] = temp

  temp = (1 + saturation / 200.0) * hlsImg[:, :, 2]
  temp[temp > 255] = 255  
  hlsImg[:, :, 2] = temp 
  result_img = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)

  face = cut_face(result_img,faceb)
  face = cv2.bilateralFilter(face, d=0, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

  blur_img = cv2.GaussianBlur(face, (sharpe, sharpe),0)
  face = cv2.addWeighted(face, 1.4, blur_img, -0.4, 0)

  result_img = bind_face(result_img,face,faceb)

  #find_face(result_img,faceb,landmark=False)
  
  # 顯示圖片
  cv2.imshow(winName, result_img)

  # 若按下 q 鍵則離開迴圈
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()