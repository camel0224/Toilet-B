import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pandas as pd
import io
import re
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'inventory' not in st.session_state:
    st.session_state.inventory = pd.DataFrame(columns=['Date', 'Brand', 'Model Number', 'Home Depot Price', 'Home Depot Link', 'Lowes Price', 'Lowes Link', 'Ferguson Price', 'Ferguson Link'])

def preprocess_image(image):
    """Preprocess the image for better OCR results."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Detect orientation
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        
        # Rotate the image if needed
        if abs(angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            rotated = thresh
        
        # Increase contrast
        enhanced = cv2.convertScaleAbs(rotated, alpha=1.5, beta=0)
        
        return enhanced
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return image

def extract_model_number(text):
    """Extract model numbers from the OCR text."""
    # Patterns for different brands
    kohler_pattern = r'(?i)(?:K-)?[0-9]{4}(?:-[0-9]+)?'
    toto_pattern = r'(?i)(?:CST|MS)[0-9]{3,4}(?:[A-Z]+)?(?:-[0-9]+)?'
    
    # Search for patterns
    kohler_matches = re.findall(kohler_pattern, text)
    toto_matches = re.findall(toto_pattern, text)
    
    # Combine and clean matches
    all_matches = kohler_matches + toto_matches
    cleaned_matches = [match.upper() for match in all_matches]
    
    return cleaned_matches if cleaned_matches else None

async def fetch_price(session, url, headers, timeout=30):
    """Fetch HTML content from URL with timeout."""
    try:
        async with session.get(url, headers=headers, timeout=timeout) as response:
            if response.status == 200:
                return await response.text()
            else:
                logger.error(f"Error fetching {url}: Status {response.status}")
                return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None

async def search_product_prices(brand, model):
    """Search for product prices across different retailers."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Construct search terms
    search_term = f"{brand} {model}"
    encoded_search = search_term.replace(' ', '+')
    
    # Define retailer URLs
    urls = {
        'Home Depot': f'https://www.homedepot.com/s/{encoded_search}',
        'Lowes': f'https://www.lowes.com/search?searchTerm={encoded_search}',
        'Ferguson': f'https://www.ferguson.com/search/{encoded_search}'
    }
    
    results = {
        'Home Depot Price': 'Not found',
        'Home Depot Link': '',
        'Lowes Price': 'Not found',
        'Lowes Link': '',
        'Ferguson Price': 'Not found',
        'Ferguson Link': ''
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for retailer, url in urls.items():
                tasks.append(fetch_price(session, url, headers))
            
            responses = await asyncio.gather(*tasks)
            
            for retailer, html in zip(urls.keys(), responses):
                if html:
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    if retailer == 'Home Depot':
                        price_elem = soup.find('span', {'class': ['price-detailed__price', 'price']})
                        if price_elem:
                            results['Home Depot Price'] = price_elem.text.strip()
                            results['Home Depot Link'] = urls['Home Depot']
                    
                    elif retailer == 'Lowes':
                        price_elem = soup.find('span', {'class': ['price', 'price-main']})
                        if price_elem:
                            results['Lowes Price'] = price_elem.text.strip()
                            results['Lowes Link'] = urls['Lowes']
                    
                    elif retailer == 'Ferguson':
                        price_elem = soup.find('span', {'class': ['price', 'product-price']})
                        if price_elem:
                            results['Ferguson Price'] = price_elem.text.strip()
                            results['Ferguson Link'] = urls['Ferguson']
    
    except Exception as e:
        logger.error(f"Error in price search: {str(e)}")
    
    return results

def save_inventory():
    """Save inventory to CSV file."""
    if not st.session_state.inventory.empty:
        st.session_state.inventory.to_csv('inventory.csv', index=False)
        return True
    return False

def load_inventory():
    """Load inventory from CSV file."""
    try:
        df = pd.read_csv('inventory.csv')
        st.session_state.inventory = df
        return True
    except:
        return False

# Main app interface
st.title('Toilet Box Scanner')

# Sidebar for inventory management
with st.sidebar:
    st.header('Inventory Management')
    if st.button('Save Inventory'):
        if save_inventory():
            st.success('Inventory saved successfully!')
        else:
            st.error('No inventory to save.')
    
    if st.button('Load Inventory'):
        if load_inventory():
            st.success('Inventory loaded successfully!')
        else:
            st.error('No inventory file found.')

# Main content
tab1, tab2 = st.tabs(['Add New Item', 'View Inventory'])

with tab1:
    st.header('Add New Item')
    
    # File uploader
    uploaded_file = st.file_uploader("Upload image of the model number", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convert uploaded file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Process image and extract text
        processed_image = preprocess_image(image)
        text = pytesseract.image_to_string(processed_image)
        
        # Extract model numbers
        model_numbers = extract_model_number(text)
        
        if model_numbers:
            st.success('Model numbers found!')
            
            # Create selection for brand and model
            brand = st.selectbox('Select Brand:', ['Kohler', 'TOTO'])
            selected_model = st.selectbox('Select Model Number:', model_numbers)
            
            if st.button('Search Prices'):
                with st.spinner('Searching for prices...'):
                    # Search for prices
                    prices = asyncio.run(search_product_prices(brand, selected_model))
                    
                    # Add to inventory
                    new_row = {
                        'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Brand': brand,
                        'Model Number': selected_model,
                        **prices
                    }
                    
                    st.session_state.inventory = pd.concat([
                        st.session_state.inventory,
                        pd.DataFrame([new_row])
                    ], ignore_index=True)
                    
                    st.success('Item added to inventory!')
        else:
            st.error('No model numbers found. Please try another image or ensure the model number is clearly visible.')

with tab2:
    st.header('Inventory')
    if not st.session_state.inventory.empty:
        st.dataframe(st.session_state.inventory)
        
        # Export options
        if st.button('Export to Excel'):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                st.session_state.inventory.to_excel(writer, index=False)
            
            st.download_button(
                label="Download Excel file",
                data=output.getvalue(),
                file_name="toilet_inventory.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info('No items in inventory yet. Add items using the "Add New Item" tab.')
