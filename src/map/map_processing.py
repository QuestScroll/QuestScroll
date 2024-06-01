import fitz     # PyMuPDF
import cv2
import numpy as np
import os

# Define filepaths for resources
pdf_path = "../pdfs/DnD_Map_Scans.pdf"
image_dir = "../images/"
test_img_path = "../images/maps_page2_1.jpeg"
warp_output_path = "../images/warped_image2.jpeg"

# Open the PDF file
pdf_document = fitz.open("../pdfs/DnD_Map_Scans.pdf")

for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_filename = f"../images/maps_page{page_num+1}_{img_index}.{image_ext}"
        with open(image_filename, "wb") as image_file:
            image_file.write(image_bytes)
        
def correct_perspective(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    orig = image.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edged = cv2.Canny(gray, 75, 200)

    # Find the contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            doc_cnts = approx
            break

    # If no contour found, return the original image
    if doc_cnts is None:
        cv2.imwrite(output_path, orig)
        return
    
    # Apply the four point transform to obtain a top-down view of the original image
    pts = doc_cnts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # Find the top-left, top-right, bottom-right, and bottom-left points
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points to obtain a top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # Save the output image
    cv2.imwrite(output_path, warped)

# Usage demo
correct_perspective(test_img_path, warp_output_path)

