#!/usr/bin/env python3
"""
Pre-process sample images for the gallery
Предобработка примеров для галереи
"""

import os
import json
import cv2
from pyimagesearch.transform import four_point_transform
from pyimagesearch.imutils import resize
from scan import DocScanner
from gemini_detector import detect_corners_with_gemini, extract_text_with_gemini

try:
    from config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = None

SAMPLES_DIR = "sample_images"
OUTPUT_DIR = "static/examples"
MANIFEST_PATH = "static/examples/manifest.json"


def preprocess_samples():
    """Process all sample images and create manifest"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    scanner = DocScanner()
    examples = []
    
    sample_files = [f for f in os.listdir(SAMPLES_DIR) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for idx, filename in enumerate(sorted(sample_files)):
        print(f"\n[{idx+1}/{len(sample_files)}] Processing {filename}...")
        
        src_path = os.path.join(SAMPLES_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        
        # Read image
        image = cv2.imread(src_path)
        if image is None:
            print(f"  ❌ Could not read {filename}")
            continue
        
        # Create thumbnail for gallery
        thumb = resize(image, height=150)
        thumb_path = os.path.join(OUTPUT_DIR, f"{base_name}_thumb.jpg")
        cv2.imwrite(thumb_path, thumb)
        
        # Copy original to static
        orig_path = os.path.join(OUTPUT_DIR, f"{base_name}_original.jpg")
        orig_resized = resize(image, height=800)
        cv2.imwrite(orig_path, orig_resized)
        
        # Process with OpenCV
        print("  → OpenCV processing...")
        ratio = image.shape[0] / 500.0
        rescaled = resize(image, height=500)
        
        try:
            screenCnt = scanner.get_contour(rescaled)
            
            # Step 2: Contour visualization
            rescaled_with_contour = rescaled.copy()
            cv2.drawContours(rescaled_with_contour, [screenCnt], -1, (0, 255, 0), 2)
            step2_opencv_path = os.path.join(OUTPUT_DIR, f"{base_name}_step2_opencv.jpg")
            cv2.imwrite(step2_opencv_path, rescaled_with_contour)
            
            # Step 3: Warped result
            warped = four_point_transform(image, screenCnt.reshape(4, 2) * ratio)
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            warped_resized = resize(warped_gray, height=650)
            result = cv2.threshold(warped_resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            step3_opencv_path = os.path.join(OUTPUT_DIR, f"{base_name}_step3_opencv.jpg")
            cv2.imwrite(step3_opencv_path, result)
            
            opencv_success = True
        except Exception as e:
            print(f"  ❌ OpenCV error: {e}")
            opencv_success = False
            step2_opencv_path = orig_path
            step3_opencv_path = orig_path
        
        # Process with Gemini (if available)
        step2_gemini_path = orig_path
        step3_gemini_path = orig_path
        extracted_text = ""
        gemini_success = False
        
        if GEMINI_API_KEY:
            print("  → Gemini processing...")
            try:
                corners = detect_corners_with_gemini(src_path, GEMINI_API_KEY)
                
                if corners is not None:
                    # Step 2: Corner visualization
                    viz_image = image.copy()
                    pts = corners.astype(int)
                    cv2.polylines(viz_image, [pts], True, (0, 0, 255), 5)
                    for p in pts:
                        cv2.circle(viz_image, tuple(p), 10, (255, 0, 0), -1)
                    viz_resized = resize(viz_image, height=500)
                    step2_gemini_path = os.path.join(OUTPUT_DIR, f"{base_name}_step2_gemini.jpg")
                    cv2.imwrite(step2_gemini_path, viz_resized)
                    
                    # Step 3: Warped result
                    warped = four_point_transform(image, corners)
                    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    warped_resized = resize(warped_gray, height=650)
                    result = cv2.threshold(warped_resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    step3_gemini_path = os.path.join(OUTPUT_DIR, f"{base_name}_step3_gemini.jpg")
                    cv2.imwrite(step3_gemini_path, result)
                    
                    gemini_success = True
                
                # Extract text
                print("  → Extracting text...")
                extracted_text = extract_text_with_gemini(src_path, GEMINI_API_KEY) or ""
                
            except Exception as e:
                print(f"  ❌ Gemini error: {e}")
        
        # Add to manifest
        examples.append({
            "id": base_name,
            "name": f"Example {idx + 1}",
            "thumbnail": f"static/examples/{base_name}_thumb.jpg",
            "original": f"static/examples/{base_name}_original.jpg",
            "opencv": {
                "step2": f"static/examples/{base_name}_step2_opencv.jpg" if opencv_success else None,
                "step3": f"static/examples/{base_name}_step3_opencv.jpg" if opencv_success else None,
            },
            "gemini": {
                "step2": f"static/examples/{base_name}_step2_gemini.jpg" if gemini_success else None,
                "step3": f"static/examples/{base_name}_step3_gemini.jpg" if gemini_success else None,
            },
            "text": extracted_text
        })
        
        print(f"  ✓ Done: {base_name}")
    
    # Save manifest
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump({"examples": examples}, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Processed {len(examples)} examples")
    print(f"📄 Manifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    preprocess_samples()
