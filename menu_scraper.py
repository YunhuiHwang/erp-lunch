import os
# Set environment variable to handle OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import requests
from bs4 import BeautifulSoup
import easyocr
import numpy as np
import datetime
import pymsteams
import re
import sys
import cv2

OUTPUT_DIR = 'ocr_text'

# Configuration
# Users should set this environment variable or edit the string below
TEAMS_WEBHOOK_URL = os.environ.get('TEAMS_WEBHOOK_URL', '') 

GTP_MENU_LIST_URL = 'https://www.gtp.or.kr/web/bbs/pdsList.jsp?gubun=weekMenu'
GTP_BASE_URL = 'https://www.gtp.or.kr'

def get_latest_menu_post_url():
    """Finds the URL of the menu post that covers today's date."""
    try:
        response = requests.get(GTP_MENU_LIST_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = soup.select('a[href*="pdsView.jsp?gubun=weekMenu"]')
        
        today = datetime.date.today()
        target_link = None
        
        print(f"Looking for menu covering: {today}")
        
        for link in links:
            title = link.get_text(strip=True)
            match = re.search(r'\((\d+)\.(\d+)\.\~(\d+)\.(\d+)\.\)', title)
            if match:
                m1, d1, m2, d2 = map(int, match.groups())
                current_year = today.year
                
                try:
                    start_date = datetime.date(current_year, m1, d1)
                    end_date = datetime.date(current_year, m2, d2)
                    
                    if m1 == 12 and m2 == 1:
                        end_date = datetime.date(current_year + 1, m2, d2)
                    if today.month == 1 and m1 == 12:
                         start_date = datetime.date(current_year - 1, m1, d1)
                         
                    print(f"Checking post '{title}' ({start_date} ~ {end_date})")
                    
                    if start_date <= today <= end_date:
                        target_link = link['href']
                        print(f"-> Found matching post!")
                        break
                except ValueError:
                    continue
        
        if not target_link:
            print("Today's menu post not found matching the date range.")
            print("Defaulting to the latest post as fallback.")
            if links:
                target_link = links[0]['href']
            else:
                return None
                
        if not target_link.startswith('http'):
            if target_link.startswith('/'):
                 target_link = GTP_BASE_URL + target_link 
            else:
                 target_link = GTP_BASE_URL + '/web/bbs/' + target_link
            
        print(f"Selected menu post URL: {target_link}")
        return target_link
    except Exception as e:
        print(f"Error fetching menu list: {e}")
        return None

def get_menu_image_url(post_url):
    """Extracts the image URL from the post page."""
    try:
        response = requests.get(post_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        images = soup.select('img')
        target_image_url = None
        
        for img in images:
            src = img.get('src', '')
            if 'upload' in src or 'pds' in src:
                target_image_url = src
                break 
        
        if target_image_url:
            if not target_image_url.startswith('http'):
                target_image_url = GTP_BASE_URL + target_image_url
            print(f"Menu image URL: {target_image_url}")
            return target_image_url
        else:
            print("No suitable image found in the post.")
            return None
            
    except Exception as e:
        print(f"Error fetching post content: {e}")
        return None

def download_image(image_url):
    """Downloads the image to a temporary file."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(image_url, stream=True, headers=headers)
        response.raise_for_status()
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        filename = os.path.join(OUTPUT_DIR, 'menu_temp.jpg')
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Image downloaded to {filename}")
        return filename
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def preprocess_image(image_path):
    """Upscales and thresholds the image to improve OCR accuracy."""
    print("Preprocessing image for better OCR...")
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
            
        # Upscale
        scale_percent = 200 # 2x
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple thresholding/denoising can sometimes help, but sometimes hurt.
        # Let's stick with just upscaling and grayscale which is usually safer for EasyOCR.
        # This helps separate text from background
        # Tuned parameters: 11, 2 (Standard)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        new_filename = os.path.join(OUTPUT_DIR, 'menu_processed.jpg')
        cv2.imwrite(new_filename, binary)
        print(f"Processed image saved to {new_filename}")
        return new_filename
    except Exception as e:
        print(f"Preprocessing failed: {e}. Using original image.")
        return image_path

def fix_typos(text):
    """Corrects common OCR errors based on a dictionary."""
    corrections = {
        '설러드': '샐러드',
        '셀러드': '샐러드',
        '각두기': '깍두기',
        '깍뚜기': '깍두기',
        '배추김치&각두기': '배추김치&깍두기',
        '배추김치&깍뚜기': '배추김치&깍두기',
        '꼬빵': '꽃빵',
        '꽂빵': '꽃빵',
        '숨눕': '숭늉',
        '숭능': '숭늉',
        '동그랑명전': '동그랑땡전',
        '동그랑평전': '동그랑땡전',
        '껏임심': '깻잎쌈',
        '째임삼': '깻잎쌈',
        '임삼': '깻잎쌈',
        '깻입': '깻잎',
        '피출': '피클',
        '할라피뇨': '할라피뇨',
        '활라피뇨': '할라피뇨',
        '돈육바베규': '돈육바베큐',
        '바베규': '바베큐',
        '물고기': '불고기',
        '포레': '포케',
        '그린설러드': '그린샐러드',
        '리코다': '리코타',
        '물고기포레': '물고기..?(확인필요)', # This one is tricky, maybe '불고기'? Leave for now or guess.
        '두부면물고기': '두부면불고기', # Guessing Bulgolgi
        '두부면': '두부면',
        '오븐구이': '오븐구이',
        '오븐구': '오븐구이',
        '도토리묵': '도토리묵'
    }
    
    clean_text = text.replace(" ", "")
    for wrong, right in corrections.items():
        if wrong in clean_text or wrong in text:
             text = text.replace(wrong, right)
             
    # Specific full-word fixes for very short matches
    if text == '숨눕': return '숭늉'
    
    return text

def get_bbox_center(bbox):
    pts = np.array(bbox)
    center = np.mean(pts, axis=0)
    return center[0], center[1]

def get_bbox_range(bbox):
    pts = np.array(bbox)
    return np.min(pts[:, 0]), np.max(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 1])

def parse_menu_with_ocr_and_get_data(image_path):
    # Preprocess first
    processed_path = preprocess_image(image_path)
    
    print("Initializing EasyOCR (this may take a moment)...")
    reader = easyocr.Reader(['ko', 'en']) 
    print("Running OCR...")
    ocr_result = reader.readtext(processed_path)
    
    print("Debug: First 10 generated text items:")
    for i, (bbox, text, prob) in enumerate(ocr_result[:10]):
        print(f"[{i}] {text} (prob: {prob:.2f})")
    print("End of Debug info")
    
    today = datetime.date.today()
    today_match_str = today.strftime("%Y-%m-%d")
    
    today_x_min = 0
    today_x_max = 0
    today_y_max = 0
    corner_a_y = 0
    corner_b_y = 0
    salad_bar_y = 0
    takeout_y = 0
    
    found_today = False
    
    print(f"Looking for date: {today_match_str}")
    for (bbox, text, prob) in ocr_result:
        clean_text = text.replace(" ", "")
        if today_match_str in clean_text:
            print(f"Found today's date: {text}")
            today_x_min, today_x_max, _, today_y_max = get_bbox_range(bbox)
            found_today = True
            break
            
    if not found_today:
        print("Could not find today's date header. Trying fuzzy match.")
        short_date = today.strftime("%m-%d")
        for (bbox, text, prob) in ocr_result:
             clean_text = text.replace(" ", "")
             if short_date in clean_text:
                print(f"Found today's date (short): {text}")
                today_x_min, today_x_max, _, today_y_max = get_bbox_range(bbox)
                found_today = True
                break
    
    if not found_today:
        print("Today's menu not found in the image headers.")
        return None

    # Define search bounds based on the date header
    search_x_min = today_x_min - 100
    search_x_max = today_x_max + 100
    
    corners = []
    
    for (bbox, text, prob) in ocr_result:
        clean_text = text.lower().replace(" ", "")
        _, _, min_y, max_y = get_bbox_range(bbox)
        
        if 'corn' in clean_text:
            cy = (min_y + max_y) / 2
            corners.append({'text': text, 'y': cy, 'min_y': min_y, 'max_y': max_y}) 
        elif 'takeout' in clean_text:
             takeout_y = min_y
        elif '샐러드바' in clean_text or '러드바' in clean_text or 'salad' in clean_text:
             if salad_bar_y == 0 or min_y < salad_bar_y:
                 salad_bar_y = min_y

    corners.sort(key=lambda k: k['y'])
    
    corner_b_label_y = 0
    if len(corners) >= 2:
        corner_b_label_y = corners[1]['y']
    else:
        if len(corners) == 1:
             corner_b_label_y = corners[0]['y'] + 600 # Adjusted for upscaled image (2x)
        else:
             corner_b_label_y = today_y_max + 800 # Adjusted for upscaled image

    # Boundaries must account for 2x scale?
    # Actually, easyocr returns coordinates based on the input image.
    # Since we passed the UPSCALED image, all coordinates are 2x.
    # corner_b_label_y is already in 2x space.
    
    section_a_top = today_y_max
    
    # 50px buffer in 1x scale -> 100px in 2x scale
    section_b_top = corner_b_label_y - 100 
    
    if section_b_top < section_a_top + 40:
        section_b_top = section_a_top + 300
    
    if salad_bar_y == 0:
        for (bbox, text, prob) in ocr_result:
            if '음료' in text or '숭늉' in text: 
                 _, _, min_y, _ = get_bbox_range(bbox)
                 salad_bar_y = min_y 
                 break
    
    if salad_bar_y == 0:
         salad_bar_y = takeout_y if takeout_y > 0 else 99999

    section_b_bottom = salad_bar_y - 20 # Buffer
    
    if takeout_y == 0:
        takeout_y = 99999
    
    menu_a = []
    menu_b = []
    menu_takeout = []
    
    column_items = []
    for (bbox, text, prob) in ocr_result:
        cx, cy = get_bbox_center(bbox)
        if search_x_min <= cx <= search_x_max:
             column_items.append((cy, text))
    column_items.sort(key=lambda x: x[0])
    
    # Adjust Take Out boundary slightly up to catch items above the label center
    takeout_boundary = takeout_y - 60 if takeout_y < 90000 else 99999

    for cy, text in column_items:
        # Typos fix
        text_fixed = fix_typos(text)
        
        if "2026" in text or "요일" in text:
             continue
        if cy < section_a_top + 20: 
             continue
            
        if section_a_top < cy < section_b_top:
            menu_a.append(text_fixed)
        elif section_b_top < cy < section_b_bottom:
            if 'Corner' in text or 'corner' in text:
                continue
            menu_b.append(text_fixed)
        elif cy > takeout_boundary: # Use adjusted boundary
            menu_takeout.append(text_fixed)
            
    return {
        "date": today.strftime("%Y-%m-%d"),
        "corner_a": menu_a,
        "corner_b": menu_b,
        "takeout": menu_takeout
    }

def format_menu_markdown(data):
    max_len = max(len(data['corner_a']), len(data['corner_b']))
    list_a = data['corner_a'] + [''] * (max_len - len(data['corner_a']))
    list_b = data['corner_b'] + [''] * (max_len - len(data['corner_b']))
    
    md_text = f"### {data['date']} 오늘의 점심 메뉴\n\n"
    md_text += "| Corner A | Corner B |\n"
    md_text += "| :--- | :--- |\n"
    for a, b in zip(list_a, list_b):
        md_text += f"| {a.strip()} | {b.strip()} |\n"
    md_text += "\n"
    
    if data['takeout']:
        md_text += "### Take Out\n"
        for item in data['takeout']:
            md_text += f"- {item}\n"
    return md_text

def send_to_teams(formatted_text, webhook_url):
    try:
        myTeamsMessage = pymsteams.connectorcard(webhook_url)
        myTeamsMessage.text(formatted_text)
        myTeamsMessage.send()
        print("Successfully sent to Teams.")
    except Exception as e:
        print(f"Failed to send to Teams: {e}")

def main():
    print("Starting GTP Menu Scraper...")
    if not TEAMS_WEBHOOK_URL:
        print("WARNING: TEAMS_WEBHOOK_URL is not set.")
    
    post_url = get_latest_menu_post_url()
    if not post_url: return
    image_url = get_menu_image_url(post_url)
    if not image_url: return
    image_path = download_image(image_url)
    if not image_path: return
    
    parsed_data = parse_menu_with_ocr_and_get_data(image_path)
    if not parsed_data: 
        print("Failed to parse menu data.")
        return
        
    final_text = format_menu_markdown(parsed_data)
    print("\n--- Generated Message Preview ---")
    print(final_text)
    print("---------------------------------\n")
    
    # Save preview to file
    preview_filename = os.path.join(OUTPUT_DIR, "menu_preview.txt")
    with open(preview_filename, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"Preview text saved to {preview_filename}")
    
    if TEAMS_WEBHOOK_URL:
        # Safety check: Ask for user confirmation before sending
        print("Teams Webhook URL이 설정되어 있습니다.")
        user_input = input("위 내용을 실제로 Teams에 전송하시겠습니까? (y/n): ")
        
        if user_input.lower().strip() == 'y':
            send_to_teams(final_text, TEAMS_WEBHOOK_URL)
        else:
            print("전송이 취소되었습니다.")
    else:
        print("Skipping Teams send (No URL).")

if __name__ == "__main__":
    main()
