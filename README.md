# Tech-Beauty-AI

Tech-Beauty-AI is a web application that leverages artificial intelligence to provide personalized makeup recommendations and virtual try-on experiences. Users can upload or capture their photos, and the app analyzes facial features to suggest suitable makeup products and styles.

---

## Features

- **AI-Powered Makeup Recommendation:** Get personalized makeup suggestions based on your facial features.
- **Virtual Try-On:** Upload or capture your photo and preview different makeup styles.
- **User-Friendly Interface:** Simple and intuitive UI for seamless user experience.
- **Secure Image Handling:** Uploaded images are processed securely and not shared with third parties.

---

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) A virtual environment tool like `venv` or `conda`

### Steps

1. **Clone the Repository**

   ```sh
   git clone https://github.com/yourusername/Tech-Beauty-AI.git
   cd Tech-Beauty-AI
   ```

2. **Create and Activate a Virtual Environment (Recommended)**

   ```sh
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **Download Model Weights**

   Ensure `modelft.pth` is present in the project root. If not, download it from the provided link or contact the maintainer.

5. **Run the Application**

   ```sh
   python app.py
   ```

6. **Access the App**

   Open your browser and navigate to `http://127.0.0.1:5000/`

---

## Project Structure

```
Tech-Beauty-AI/
│
├── app.py                # Main Flask application
├── makeup_app.py         # Makeup recommendation and processing logic
├── modelft.pth           # Pre-trained AI model weights
├── requirements.txt      # Python dependencies
├── static/               # Static files (CSS, JS, images)
│   ├── css/
│   ├── img/
│   └── js/
├── templates/
│   └── index.html        # Main HTML template
└── uploads/              # Uploaded user images
```

---

## How It Works

1. **User Interaction**
   - The user visits the web interface and uploads a photo or captures one using their device camera.
   - The image is sent to the backend for processing.

2. **Backend Processing**
   - The Flask backend (`app.py`) receives the image and saves it in the `uploads/` directory.
   - The image is passed to the AI model (loaded from `modelft.pth` via `makeup_app.py`).
   - The model analyzes facial features and generates personalized makeup recommendations.

3. **Recommendation & Virtual Try-On**
   - The backend returns recommended makeup styles and products.
   - The frontend (JavaScript in `static/js/main.js`) displays the recommendations and overlays virtual makeup on the user's photo for preview.

4. **Result Display**
   - Users can view, adjust, and try different makeup types (lipstick, eyeshadow, etc.) directly in the browser.

---

## How to Update Product Links

You can easily update the links to makeup products shown in the recommendations, even if you are not a technical person. Here’s how:

1. **Locate the Product Links File:**
   - Product links are usually stored in the HTML template (`index.html`).
   - If you are unsure, ask the developer or look for a file that contains website links (they often start with `http://` or `https://`).

2. **Open the File:**
   - Double-click the file in your project folder to open it.
   - Here I have arranged links in HTML like this:
     ```html
     <button class="buynow-btn" onclick="window.open('https://www.reneecosmetics.in/products/renee-peachy-pink-blush-duo', '_blank')">Peachy</button>
     ```

3. **Update the Links:**
   - To change a link, simply replace the old URL with your new product link.
   - For example, change:
   - In HTML:
     ```html
     <button class="buynow-btn" onclick="window.open('https://changed-link/', '_blank')">Product</button>
     ```

4. **Save the File:**
   - After updating, save the file (press `Ctrl + S`).

5. **Restart the App:**
   - If the app is running, stop it and start it again to see your changes.

**Tip:**  
If you need help finding the right file or line, search for the product name (like "Renee-Lipstick-Shade") in your code editor.

---

## Customization

- **Adding New Makeup Styles:** Update the AI model and frontend assets.
- **Styling:** Modify `static/css/style.css` for UI changes.
- **Model Improvements:** Replace `modelft.pth` with a new trained model for better recommendations.

---

### ⚠️ Color Format Handling (RGB vs BGR)

This project involves integration with OpenCV, which expects colors in **BGR format**, while standard web technologies (like CSS and HTML) use **RGB format**. To maintain consistency and avoid bugs, the following convention is used throughout the codebase:

- **HTML buttons** (`.color-btn`) store color values in the `data-color` attribute using **RGB** format (e.g., `[234,178,147]`), which aligns with browser rendering.
- The **`recommendations` object** also uses colors in **RGB** format for readability and direct compatibility with the UI.
- Internally, within the JavaScript logic, colors are **converted to BGR** format and stored in the `makeupOptions` object. This BGR array is then passed to any OpenCV-dependent functions or modules that require it.

This consistent separation ensures the UI functions correctly in RGB, while OpenCV receives the proper BGR format it expects.
