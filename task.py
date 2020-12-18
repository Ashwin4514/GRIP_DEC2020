import cv2
import pytesseract
from pytesseract import Output

#Mentioning env variable to use tesseract engine for OCR.
pytesseract.pytesseract.tesseract_cmd= r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#Reading the Image and resizing it 
image= cv2.imread("C:/Users/ASHWIN/Desktop/Text-Mining-01-1200x900.jpg")


#preprocessing the image to detect text from that image.
image_gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#comverting the image to grayscale and then removing noise using gaussian blur.
#using Otsu's binarization
#since we don't know the right threshold to process our image with.
image_gray= cv2.GaussianBlur(image_gray,(5,5),1)

r,T1 = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

#tesseractboxes: Making boxes around the text detected.
d= pytesseract.image_to_data(image_gray,output_type=Output.DICT)
nb= len(d['level'])
for i in range(nb):
     if(d['text'][i] != ""):
        (x,y,w,h)= (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(image_gray, (x,y),(x+w,y+h), (0,255,0),2)

#Showing the Original Image
image= cv2.resize(image,(500,500),fx=0.5,fy=0.5)        
cv2.imshow("Original Image",image)

#Showing the image with detected boxes.
image_gray_1= cv2.resize(image_gray,(500,500),fx=0.5,fy=0.5)      
cv2.imshow("image_gray",image_gray_1)
#printing the detected text
text= pytesseract.image_to_string(image_gray)
print(text)
cv2.waitKey(0)


