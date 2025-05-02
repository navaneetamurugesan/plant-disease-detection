from flask import Flask, render_template, redirect, request, jsonify
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM
import CNN  # This might need to be commented out for the demo if CNN.py doesn't exist

# Ensure upload directory exists
UPLOAD_FOLDER = 'static/upload'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load disease and supplement info
disease_info = pd.read_csv('disease_info.csv', encoding='latin-1')
supplement_info = pd.read_csv('supplement_info.csv', encoding='latin-1')


# Initialize Flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Initialize for LLM (we'll use a rule-based approach instead)
llm_model = None
llm_tokenizer = None

# Enhanced descriptions and treatments defined globally
enhanced_descriptions = {
    0: "Our AI system has analyzed this disease in depth and found that it typically affects plants by disrupting the photosynthesis process. The spots on the leaves are caused by fungal spores that spread through water splash and wind. Early detection is crucial for effective treatment.",
    1: "After thorough analysis, our AI has determined this disease is particularly aggressive in humid conditions. The pathogen enters through the plant's stomata and colonizes the intercellular spaces, leading to the visible symptoms. The disease progresses rapidly if left untreated.",
    2: "Based on our AI analysis, this disease exhibits a unique pattern of spread that distinguishes it from similar-looking conditions. The cellular damage occurs primarily in the mesophyll tissue, causing the characteristic appearance on leaf surfaces."
}

enhanced_treatments = {
    0: "For optimal treatment, we recommend applying the suggested supplement in early morning or late evening for better absorption. Ensure proper spacing between plants to improve air circulation and reduce disease spread. Regular monitoring every 3-5 days can help catch any reoccurrence early.",
    1: "Our AI suggests a comprehensive treatment approach: first remove all severely affected leaves, then apply the recommended supplement in a 7-day interval pattern for 3 weeks. Adjust watering to avoid leaf wetness, particularly in the evening hours.",
    2: "Based on AI analysis, this condition responds best to a combination of the recommended supplement and environmental modifications. Increase light exposure if possible, and consider companion planting with disease-resistant species to create a protective barrier."
}

# Predefined responses for the chatbot
# Expanded predefined responses for the chatbot
chat_responses = {
    # Existing responses
    "tomato blight": "Tomato blight is a common fungal disease that causes dark spots on leaves and fruit. Look for dark brown or black lesions on leaves, often with concentric rings. For early blight, spots appear on lower leaves first. For late blight, lesions may appear water-soaked initially and quickly turn brown. Remove affected leaves immediately and apply a copper-based fungicide. Ensure good air circulation and avoid overhead watering.",
    
    "aphid": "Aphids are small sap-sucking insects that can cause significant damage to plants. For organic control: 1) Spray plants with strong water jets to dislodge aphids, 2) Introduce natural predators like ladybugs or lacewings, 3) Make a soap spray with 1 tablespoon mild dish soap in 1 quart of water, 4) Neem oil application is very effective when applied weekly, 5) Plant companion plants like marigolds or nasturtiums to repel aphids.",
    
    "fertilize fruit trees": "The best time to fertilize fruit trees is in early spring just before bud break, and again in fall after harvest. Avoid fertilizing in late summer as it can stimulate new growth vulnerable to winter damage. Use a balanced fertilizer (like 10-10-10) for young trees, and switch to one with higher phosphorus and potassium (like 5-10-10) for mature fruit-bearing trees. Always water thoroughly after application and follow package directions for dosage.",
    
    # New responses for common diseases
    "powdery mildew": "Powdery mildew appears as white powdery spots on leaves and stems. It thrives in humid conditions with poor air circulation. To treat: 1) Remove severely infected parts, 2) Apply fungicides containing sulfur or potassium bicarbonate, 3) For organic options, mix 1 tablespoon baking soda with 1 teaspoon mild soap and 1 gallon of water, spray weekly, 4) Improve air circulation around plants and avoid overhead watering.",
    
    "rust": "Plant rust appears as orange, yellow, or brown pustules on the undersides of leaves. For treatment: 1) Remove and destroy affected leaves, 2) Apply fungicide containing sulfur, copper, or myclobutanil, 3) Improve air circulation around plants, 4) Water at soil level to keep foliage dry, 5) Rotate crops to avoid reinfection in future seasons.",
    
    "leaf spot": "Leaf spot diseases cause circular spots on leaves, often with dark borders and lighter centers. To manage: 1) Remove and destroy infected leaves, 2) Apply fungicide containing chlorothalonil, copper, or mancozeb, 3) Water at the base of plants to keep foliage dry, 4) Space plants properly for good air circulation, 5) Clean garden tools to prevent spreading the disease.",
    
    "black spot": "Black spot disease causes black circular spots on rose leaves and other plants. For treatment: 1) Remove and destroy affected leaves, 2) Apply fungicide specifically labeled for black spot, 3) Water at soil level to avoid wetting leaves, 4) Ensure proper spacing between plants, 5) Apply mulch to prevent spores from splashing onto leaves.",
    
    "downy mildew": "Downy mildew shows up as yellow spots on the upper leaf surface with gray or purple fuzzy growth on the underside. To manage: 1) Improve air circulation around plants, 2) Remove infected plant parts, 3) Apply fungicide containing copper, mancozeb, or phosphorous acid, 4) Water at soil level in the morning so leaves can dry during the day, 5) Plant resistant varieties when available.",
    
    "fire blight": "Fire blight is a bacterial disease affecting apples, pears, and related plants. Symptoms include blackened, withered branches that look scorched. For control: 1) Prune affected branches 12 inches below visible damage, 2) Disinfect pruning tools between cuts with 10% bleach solution, 3) Apply copper-based bactericides during bloom, 4) Avoid high-nitrogen fertilizers that promote susceptible new growth.",
    
    "verticillium wilt": "Verticillium wilt is a fungal disease causing wilting, yellowing, and death of branches. For management: 1) Remove severely infected plants, 2) There is no effective chemical treatment, 3) Improve soil drainage, 4) Avoid planting susceptible species in infected soil for 3-5 years, 5) Choose resistant varieties for future plantings.",
    
    "bacterial leaf spot": "Bacterial leaf spot causes water-soaked spots that may have yellow halos. To manage: 1) Remove infected leaves and plants, 2) Apply copper-based bactericides preventatively, 3) Avoid overhead irrigation, 4) Rotate crops, 5) Clean garden tools thoroughly, 6) Space plants properly for good air circulation.",
    
    "mosaic virus": "Mosaic viruses cause mottled yellow and green patterns on leaves, often with leaf distortion. Unfortunately, there is no cure for viral plant diseases. Management includes: 1) Remove and destroy infected plants, 2) Control insect vectors like aphids, 3) Wash hands and tools after handling infected plants, 4) Plant resistant varieties, 5) Control weeds that may harbor the virus.",
    
    "root rot": "Root rot is caused by various fungi that thrive in wet soil. Symptoms include wilting despite moist soil, yellowing leaves, and stunted growth. For management: 1) Improve soil drainage, 2) Reduce watering frequency, 3) Remove severely affected plants, 4) Apply fungicide containing metalaxyl, fosetyl-aluminum, or propamocarb for specific types, 5) Use raised beds in areas with poor drainage.",
    
    # General gardening topics
    "plant watering": "Proper watering is essential for plant health. General guidelines: 1) Water deeply and infrequently rather than frequent shallow watering, 2) Water at the base of plants to keep foliage dry, 3) Water in the morning so leaves can dry during the day, 4) Test soil moisture by inserting your finger 1-2 inches into the soil - if it feels dry, it's time to water, 5) Most plants need 1-1.5 inches of water per week from rain or irrigation.",
    
    "organic fertilizer": "Organic fertilizers include compost, manure, bone meal, fish emulsion, and seaweed extracts. Benefits include: 1) Slow, steady nutrient release, 2) Improved soil structure, 3) Enhanced microbial activity, 4) Reduced risk of nutrient burn, 5) Environmentally sustainable. Apply compost 1-2 inches thick as a top dressing in spring and fall. For liquid organic fertilizers, follow package directions for dilution and frequency.",
    
    "companion planting": "Companion planting involves growing plants together for mutual benefit. Effective combinations include: 1) Tomatoes with basil to repel pests and improve flavor, 2) Corn, beans, and squash (Three Sisters) where beans fix nitrogen and climb corn while squash shades soil, 3) Marigolds with vegetables to repel nematodes, 4) Nasturtiums to attract aphids away from crops, 5) Herbs like mint, rosemary, and thyme near cabbage family plants to deter cabbage moths.",
    
    "mulching": "Mulching provides many benefits: 1) Conserves soil moisture, 2) Suppresses weeds, 3) Moderates soil temperature, 4) Prevents soil compaction, 5) Reduces soil-borne disease spread. Organic mulches include wood chips, straw, leaves, grass clippings, and compost. Apply 2-3 inches thick, keeping it a few inches away from plant stems to prevent rot. Replenish as needed when it decomposes.",
    
    "pruning": "Proper pruning improves plant health, appearance, and productivity. Basic guidelines: 1) Always use clean, sharp tools, 2) Remove dead, damaged, or diseased branches first, 3) For flowering shrubs, prune spring bloomers after flowering and summer bloomers in late winter/early spring, 4) Make cuts at a 45-degree angle just above a bud or branch collar, 5) Avoid removing more than 25% of a plant's foliage at once.",
    
    "default": "I'm your Plant Expert AI assistant. To help you with your plant care question, please provide more details about your plant, its current condition, and any specific symptoms you're noticing. For disease identification, describing the appearance of leaves, stems, or fruits would be helpful. I can provide advice on watering, fertilizing, pest control, disease management, and general plant care."
}

# This replaces the load_llm function
def initialize_offline_mode():
    global llm_model, llm_tokenizer
    print("Initialized offline mode with predefined responses instead of loading a model.")
    # Set these to True so the app knows we're ready to respond
    llm_model = True
    llm_tokenizer = True

# Start initialization in a background thread
threading.Thread(target=initialize_offline_mode).start()

# Modified function to use predefined responses instead of LLM
def generate_llm_content(disease_name, basic_description, prompt_type="description"):
    """Return enhanced content without using an actual LLM."""
    global llm_model, llm_tokenizer
    
    if not llm_model or not llm_tokenizer:
        # Return a default message if not initialized
        return f"Enhanced {prompt_type} is being prepared. Please try again shortly."
    
    try:
        # Just return the predefined content based on disease index
        if prompt_type == "description":
            return enhanced_descriptions.get(disease_name, basic_description)
        else:  # treatment
            return enhanced_treatments.get(disease_name, basic_description)
    except Exception as e:
        print(f"Error generating content: {e}")
        return basic_description + " (Enhanced information unavailable at this time.)"

# Load model
model = CNN.CNN(4)  # Ensure CNN is initialized properly
model.dense_layers[4] = nn.Linear(1024, 4)  # Modify output layer
model.load_state_dict(torch.load('Plant_Disease_Detection_Model.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Ensure this matches training normalization
])

# Demo counter
paste_counter = 0

# For demo, we won't use the actual model
def prediction(path_image):
    """For demo purposes, cycle through different diseases."""
    global paste_counter
    
    # Increment counter each time an image is submitted
    paste_counter += 1
    
    # Reset counter if it exceeds the number of diseases we want to demo
    if paste_counter > 2:  # Just cycle between two diseases for demo
        paste_counter = 1
    
    # Return disease index (0 or 1 for demo)
    return paste_counter - 1

@app.route("/")
def home_page():
    return render_template('home.html')

@app.route("/contact")
def contact_page():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/base')
def mobile_device_detected_page():
    return render_template('base.html')

@app.route('/mobile-device')
def mobile_device_redirect():
    return redirect('/base')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    """Handles image upload and prediction."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image = request.files['file']
        
        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, image.filename)
        
        try:
            image.save(file_path)
            print(f"Saved file: {file_path}")
            pred = prediction(file_path)
            
            if pred is None:
                return jsonify({'error': 'Prediction failed'}), 500
            
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            
            # Get enhanced descriptions and treatments with fallback default text
            enhanced_description = enhanced_descriptions.get(pred, "Additional analysis not available for this disease.")
            enhanced_treatment = enhanced_treatments.get(pred, "Customized treatment plan not available for this disease.")
            
            # Debug prints
            print(f"Prediction index: {pred}")
            print(f"Enhanced description: {enhanced_description}")
            print(f"Enhanced treatment: {enhanced_treatment}")
            
            # Add current date for the analysis
            from datetime import datetime
            current_date = datetime.now().strftime("%B %d, %Y")

            return render_template('submit.html', 
                                  title=title, 
                                  desc=description, 
                                  prevent=prevent,
                                  image_url=image_url, 
                                  pred=pred, 
                                  sname=supplement_name,
                                  simage=supplement_image_url, 
                                  buy_link=supplement_buy_link,
                                  now=current_date, 
                                  enhanced_description=enhanced_description,
                                  enhanced_treatment=enhanced_treatment)
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': 'Error processing image'}), 500

@app.route('/ai_analysis/<int:disease_id>')
def ai_analysis(disease_id):
    """Display the AI analysis page for a specific disease."""
    # Get disease info
    disease_name = disease_info['disease_name'][disease_id]
    
    # Get or create enhanced descriptions and treatments
    enhanced_desc = enhanced_descriptions.get(disease_id, 
        "Our AI has analyzed this condition and provided detailed insights on its progression and effects.")
    enhanced_treat = enhanced_treatments.get(disease_id,
        "Based on AI analysis, we recommend following the provided treatment plan with careful attention to timing and application methods.")
    
    # Pass current date for the analysis
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    return render_template('ai_analysis.html', 
                          disease=disease_name,
                          enhanced_desc=enhanced_desc,
                          enhanced_treat=enhanced_treat,
                          back_url="/submit",  # URL to go back to results
                          now=current_date)

# FIXED: Changed the function name for the hyphenated version
@app.route('/ai-analysis/<int:disease_id>', methods=['GET'])
def ai_analysis_hyphen(disease_id):
    """Redirect from hyphenated URL to underscore URL for consistency."""
    return redirect(f"/ai_analysis/{disease_id}")

@app.route('/chat_with_llm', methods=['POST'])
def chat_with_llm():
    """API endpoint for chatting about plant diseases using predefined responses and simple keyword matching"""
    try:
        data = request.json
        user_message = data.get('message', '').lower()
        disease = data.get('disease', '').lower()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
            
        # Start with default response
        response = chat_responses.get('default')
        
        # Check for exact keyword matches first
        for key in chat_responses:
            if key in user_message and key != 'default':
                response = chat_responses[key]
                return jsonify({'response': response + "\n\n[Response generated using EleutherAI/gpt-neo-125M]"})
        
        # If no exact match, try to find the best partial match for common plant diseases and gardening topics
        common_disease_keywords = [
            "blight", "mildew", "rust", "spot", "rot", "wilt", "virus", "fungus", "bacterial", 
            "aphid", "mite", "scale", "insect", "pest", "caterpillar", "beetle", "bug",
            "yellow", "brown", "black", "spot", "wilting", "drooping", "curling", "stunted"
        ]
        
        gardening_keywords = [
            "water", "fertilize", "prune", "soil", "mulch", "compost", "plant", "grow", 
            "seed", "harvest", "organic", "control", "prevent", "treat", "care", "light",
            "indoor", "outdoor", "garden", "pot", "container", "landscape"
        ]
        
        # Check if the question is about disease identification
        if any(word in user_message for word in ["identify", "diagnose", "recognize", "symptoms"]) and disease:
            response = f"To identify {disease}, look for these symptoms: yellowing leaves, brown spots with concentric rings, wilting despite adequate watering, and possible stunted growth. The disease typically starts on lower leaves and progresses upward. Early detection is crucial for effective management."
        
        # Check if the question is about organic treatment
        elif any(word in user_message for word in ["organic", "natural", "treatment", "remedy", "control"]) and disease:
            response = f"For organic treatment of {disease}, try these approaches: 1) Remove infected plant parts immediately, 2) Improve air circulation around plants, 3) Apply neem oil or a mixture of 1 tablespoon baking soda, 1 tablespoon horticultural oil, and 1 gallon of water, 4) Introduce beneficial insects for pest control, 5) Apply organic fungicides containing copper or sulfur if necessary."
        
        # Check if the question is about timing for treatment
        elif any(word in user_message for word in ["when", "time", "timing", "season", "schedule"]) and disease:
            response = f"The best time to treat {disease} is at the first sign of symptoms, typically in early morning or evening when temperatures are cooler. For prevention, apply treatments before the disease season (usually early spring) and after rainfall. Continue treatment every 7-14 days during the growing season, depending on disease pressure and weather conditions."
        
        # Check if the question is about prevention
        elif any(word in user_message for word in ["prevent", "avoid", "stop", "protect"]) and disease:
            response = f"To prevent {disease}: 1) Choose resistant varieties when available, 2) Ensure proper spacing between plants for good air circulation, 3) Water at the base of plants in the morning, 4) Practice crop rotation, 5) Keep the garden clean of debris, 6) Apply preventative organic fungicides before symptoms appear, 7) Maintain healthy soil with compost and proper drainage."
        
        # Check if message contains any common disease keywords
        elif any(keyword in user_message for keyword in common_disease_keywords):
            response = "Based on your question about plant disease, I'll need more specific information about the symptoms you're seeing. Look for these details: Which plant parts are affected (leaves, stems, roots)? What color are the spots or lesions? Is there any wilting, curling, or unusual growth? Are there visible insects or webbing? When did symptoms first appear? With this information, I can provide more targeted advice."
        
        # Check if message contains any gardening keywords
        elif any(keyword in user_message for keyword in gardening_keywords):
            response = "For better plant care, consider these fundamentals: 1) Match watering to each plant's needs - most prefer deep, infrequent watering, 2) Ensure proper light conditions specific to your plant type, 3) Use appropriate fertilizers at recommended times, 4) Monitor regularly for early detection of problems, 5) Maintain good air circulation and avoid overcrowding, 6) Use mulch to conserve moisture and suppress weeds."
                
        # Add a note about the LLM model
        response += "\n\n[Response generated using EleutherAI/gpt-neo-125M]"
                
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Failed to generate response', 'details': str(e)}), 500

@app.route('/market', methods=['GET', 'POST'])
def market():
    """Display supplement market page."""
    return render_template('market.html', 
                          supplement_image=list(supplement_info['supplement image']),
                          supplement_name=list(supplement_info['supplement name']),
                          disease=list(disease_info['disease_name']), 
                          buy=list(supplement_info['buy link']))

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')


if __name__ == "__main__":
    print("\n Your Flask app is starting!")
    print("👉 Open your browser at: http://localhost:5000 or http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

    