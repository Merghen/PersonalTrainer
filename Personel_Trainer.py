import numpy as np
import cv2
import math
import mediapipe as mp

# video okuma
cap=cv2.VideoCapture('spor1.mp4')

# vücutu okucak sınıf
my_pose = mp.solutions.pose

# vucutum detect edecek bir fonksiyon oluşturdum
body=my_pose.Pose()

# vücutun çizimi
draw=mp.solutions.drawing_utils

miktar=0
hareket=False # yaptı yapmadı


# video 1 için şınav tespiti.
def sinav_say(eklem_infos, ayrinti_goster=False):
    
       omuz=eklem_infos[11]
       dirsek=eklem_infos[13]
       bilek=eklem_infos[15]
       # omuz[2] -> omuz ekleminin y koordinatını verir [1] ise x koordinatını verir.
       
         
       # açıyı hesaplama
       """
          dirsek açısını bulabilmek için 3 noktaya ihtiyacım var. Omuz ile dirsek arasındaki uzaklığını
          dirsek ile bilek arasındaki uzaklıktan çıkaracağız. Bunun için. (omuzY-dirsekY,omuzX-dirsekX)
          bu bana omuz ile dirsek arasındaki uzaklığı verir - (dirsekY-bilekY,dirsekX-bilekX)
          Kısaca : (omuzY-dirsekY,omuzX-dirsekX)-(dirsekY-bilekY,dirsekX-bilekX)
       """
       aci=math.degrees(math.atan2(omuz[2]-dirsek[2],omuz[1]-dirsek[1])-math.atan2(dirsek[2]-bilek[2],dirsek[1]-bilek[1]))
       
       #aci değerleri - değer geliyor her değeri 360 artırdık ve + bir değer elde ettik...
       if aci<360:
           aci+=360
      
       
       
       # şınav pozisonun sayılması
       """
       eğer açı derecesi 295'in altına iniyorsa kişi şınavın yarısını bitirdi. bu yüzden yarım ekledik.
       aöma bir if daha koymamız lazım eğer koymazsal aci 295'in altında oldugu sürece sürekli 0.5 eklicek
       bunu kontrol etmemiz gerek. eğer kişi 295 derece altına girdiyse o hareketi yaptı olarak değiş
       ve 0.5 artır. daha sonra harektin ikinci aşaması eğer açı 350'den büyükse kişi hareketi tamamen
       bitirdi ama bir kontrol daha eklememiz lazım sürekli 0.5 artımrmaması için. Aynı mantıkla kişi
       nin dirsek açoısı 350 dem büyükse ve yarım hareketi yaptı ise 0.5 ekle ve ardomdam hareket yapıp
       yapmadığıını sıfırla. yani False yap. döngü şekline bı devam eder..
       
       """
       global hareket
       global miktar 
       if(aci<295):
           if (hareket==False):
               miktar+=0.5
               hareket=True
       if(aci>350):
           if(hareket==True):
               miktar+=0.5
               hareket=False
       print(miktar)    
       
       
       if(ayrinti_goster==True):
           
           # dirsek kısmını mavi yuvarlak
           cv2.circle(frame, (dirsek[1],dirsek[2]), 13, (0,0,255))
           cv2.circle(frame, (dirsek[1],dirsek[2]), 9, (255,0,0), cv2.FILLED)
           
           # omuz kısmını mavi yuvarlak
           cv2.circle(frame, (omuz[1],omuz[2]), 13, (0,0,255))
           cv2.circle(frame, (omuz[1],omuz[2]), 9, (255,0,0), cv2.FILLED)
           
           # bilek kısmını mavi yuvarlak
           cv2.circle(frame, (bilek[1],bilek[2]), 13, (0,0,255))
           cv2.circle(frame, (bilek[1],bilek[2]), 9, (255,0,0), cv2.FILLED)
           
           # düz çizgi omuz ile dirsek arası
           cv2.line(frame, (dirsek[1],dirsek[2]), (omuz[1],omuz[2]), (0,255,0),3)
           # düz çizgi diz ile bilek arası
           cv2.line(frame, (dirsek[1],dirsek[2]), (bilek[1],bilek[2]), (0,255,0),3)

           # açıyı yazdırdık
           cv2.putText(frame, str(int(aci)), (dirsek[1],dirsek[2]), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0))






# video 2(spor 2) için hareketi kaç defa yapmış 
def dizSay(eklem_info):
    
    omuz=eklem_info[11]
    bel=eklem_info[23]
    diz=eklem_info[25]
    
    aci2=math.degrees(math.atan2(omuz[2]-bel[2],omuz[1]-bel[1])-math.atan2(bel[2]-diz[2],bel[1]-diz[1]))
     
     #aci değerleri - değer geliyor her değeri 360 artırdık ve + bir değer elde ettik...
    if aci2<360:
         aci2+=360
      # açıyı yazdırdık
    cv2.putText(frame, str(int(aci2)), (bel[1],bel[2]), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0))
     
    
    global hareket
    global miktar 
    if(aci2<300):
        if (hareket==False):
             miktar+=0.5
             hareket=True
    if(aci2>340):
        if(hareket==True):
             miktar+=0.5
             hareket=False
    print(miktar)   
     
    

while(True):
   error,frame = cap.read()
   
   frameRgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
   #open cv bgr color setinde işlem yaıyor. biz rgb renk modeline göre detect işlemini gerçekleştirsin
   result=body.process(frameRgb)
   print(result.pose_landmarks)
   
   
   eklem_info=[]
   # eğer detect edebildiyse
   if(result.pose_landmarks!=None):
       # eklem kısımlarını çizdir
       draw.draw_landmarks(frame, result.pose_landmarks, my_pose.POSE_CONNECTIONS)
       
       for eklem,konum in  enumerate(result.pose_landmarks.landmark):
           
           # konumları düzeltelim.
           height,width,channel=frame.shape
           realX,realY=int(konum.x*width), int(konum.y*height)
          
           eklem_info.append([eklem,realX,realY])
       #print(eklem_info)    
       omuz=eklem_info[11]
       dirsek=eklem_info[13]
       bilek=eklem_info[15]
       bel=eklem_info[23]
       diz=eklem_info[25]
       
       print(omuz)

       sinav_say(eklem_infos=eklem_info,ayrinti_goster=True)
       
       #dizSay(eklem_info=eklem_info)
       

       # kaç tane şınav çekmiş ekrana yazdır: channel'ı bgr not rgb...
       cv2.putText(frame, str(int(miktar)), (50,90), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255))
    
   if (cv2.waitKey(1) &0xFF == 27): #  eğer esc tuşuna basarsa döngüden çıksın.
       break
   
   
   cv2.imshow('video', frame)


cv2.destroyAllWindows() # pencelereri kapat.
cap.release() # captur'ile işim bitti.    