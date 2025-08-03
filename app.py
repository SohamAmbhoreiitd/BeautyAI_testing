import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import base64
import cv2
import numpy as np
from makeup_app import MakeupApplication
import logging
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazily initialize the MakeupApplication class
makeup_app = None

def get_makeup_app():
    global makeup_app
    if makeup_app is None:
        logger.info("Initializing MakeupApplication...")
        makeup_app = MakeupApplication()
        logger.info("MakeupApplication initialized.")
    return makeup_app

# --- Recommended Change: Combine API endpoints ---

@app.route('/apply_makeup', methods=['POST'])
def apply_makeup():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Check if skin tone analysis should be run
        run_analysis = data.get('run_analysis', False)

        # Decode the image once
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")

        makeup_instance = get_makeup_app()
        
        # Prepare the response dictionary
        response_data = {'success': True}

        # --- Skin Analysis and Recommendation Logic (from the old /recommend) ---
        if run_analysis:
            logger.info("Running skin tone analysis...")
            skin_result = makeup_instance.extract_skin_color(image)
            
            if not skin_result['success']:
                return jsonify({'success': False, 'error': skin_result['error']}), 400
            
            skin_type = skin_result['skin_type']
            recommendations = makeup_instance.get_recommended_makeup_colors(skin_type)
            
            # Format recommendations and add to the response
            formatted_recommendations = {}
            for category, colors in recommendations.items():
                formatted_recommendations[category] = [
                    {'rgb': color, 'bgr': [color[2], color[1], color[0]], 'hex': '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])}
                    for color in colors
                ]
            
            response_data['skin_type'] = skin_type
            response_data['recommendations'] = formatted_recommendations
            logger.info(f"Analysis complete. Skin type: {skin_type}")

        # --- Makeup Application Logic ---
        makeup_options = data.get('makeup_options', {})
        
        processed_image = makeup_instance.process_frame(image, makeup_options)
        
        if processed_image is None:
            raise ValueError("Failed to process image with makeup")

        # Encode the final image to base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_data = base64.b64encode(buffer).decode('utf-8')
        
        response_data['processed_image'] = f'data:image/jpeg;base64,{processed_image_data}'
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in apply_makeup: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500

# The '/recommend' endpoint is no longer needed and can be removed.
# The other routes (index, health_check, etc.) are fine.

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health_check')
def health_check_endpoint():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)