import fitz     # PyMuPDF
import cv2
import numpy as np

# Define filepaths for resources
pdf_path = "../pdfs/DnD_Map_Scans.pdf"
image_dir = "../images/"
test_img_path = "../images/maps_page1_1.jpeg"
warp_output_path = "../images/new_warped_image1.jpeg"

# # Open the PDF file
# pdf_document = fitz.open("../pdfs/DnD_Map_Scans.pdf")

# for page_num in range(len(pdf_document)):
#     page = pdf_document.load_page(page_num)
#     image_list = page.get_images(full=True)
#     for img_index, img in enumerate(image_list, start=1):
#         xref = img[0]
#         base_image = pdf_document.extract_image(xref)
#         image_bytes = base_image["image"]
#         image_ext = base_image["ext"]
#         image_filename = f"../images/maps_page{page_num+1}_{img_index}.{image_ext}"
#         with open(image_filename, "wb") as image_file:
#             image_file.write(image_bytes)
        
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

    doc_cnts = None
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
    maxWidth = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
    maxHeight = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))

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

def detect_edges(image_path):
    # Load the perspective corrected image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform edge detection
    edged = cv2.Canny(gray, 50, 200)
    cv2.imwrite('../images/edges_detected.jpeg', edged)     # Optional: save or view the edge-detected img
    
    # We'll analyze the first and last few rows and columns to find where the white border stops
    # Check the first 10 and last 10 rows/columns
    edge_thickness = 5
    top_edge = edged[:edge_thickness, :]
    bottom_edge = edged[-edge_thickness:, :]
    left_edge = edged[:, :edge_thickness]
    right_edge = edged[:, -edge_thickness:]

    # Summing up the values in rows/columns can give us an indication of where edges might be
    top_sum = np.sum(top_edge, axis=0)
    bottom_sum = np.sum(bottom_edge, axis=0)
    left_sum = np.sum(left_edge, axis=1)
    right_sum = np.sum(right_edge, axis=1)

    # Function to determine the significant edge boundary
    def find_edge_boundary(edge_sum, threshold=50):
        edge_locs = np.where(edge_sum > threshold)[0]
        if len(edge_locs) > 0:
            return edge_locs[0], edge_locs[-1]
        return 0, len(edge_sum) - 1
    
    # Calculate boundaries where the significant changes in the edge sums occur
    top_start, top_end = find_edge_boundary(top_sum)
    bottom_start, bottom_end = find_edge_boundary(bottom_sum)
    left_start, left_end = find_edge_boundary(left_sum)
    right_start, right_end = find_edge_boundary(right_sum)

    # Output the findings
    print(f"Top Edge Boundary: Start {top_start}, End {top_end}")
    print(f"Bottom Edge Boundary: Start {bottom_start}, End {bottom_end}")
    print(f"Left Edge Boundary: Start {left_start}, End {left_end}")
    print(f"Right Edge Boundary: Start {right_start}, End {right_end}")

    # Return this information for potential use in adjustments
    return {
        "top": (top_start, top_end),
        "bottom": (bottom_start, bottom_end),
        "left": (left_start, left_end),
        "right": (right_start, right_end)
    }

# Usage demo
detect_edges("../images/warped_image.jpeg")

