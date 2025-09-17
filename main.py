from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
from typing import Dict, Any
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Garbage Classification API",
    description="FastAPI backend for garbage classification using TensorFlow",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "https://semicolon-garbage-classifier.netlify.app",
        "http://127.0.0.1:5500"
    ],
    # allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

# Class names based on your training
CLASS_NAMES = ['plastic', 'metal', 'paper']

# Image preprocessing parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Recycling recommendations
RECOMMENDATIONS = {
    'plastic': [
        'Clean the plastic item thoroughly before recycling',
        'Check the recycling number (1-7) on the bottom',
        'Remove caps and labels when possible',
        'Place in designated plastic recycling bin',
        'Avoid contaminating with food waste'
    ],
    'metal': [
        'Rinse clean of any food residue',
        'Remove paper labels when possible',
        'Aluminum cans are 100% recyclable',
        'Steel cans are magnetic and highly recyclable',
        'Place in metal recycling bin'
    ],
    'paper': [
        'Keep paper clean and dry',
        'Remove any plastic coating or tape',
        'Flatten cardboard boxes to save space',
        'Avoid wax-coated papers in regular recycling',
        'Place in paper recycling bin'
    ]
}

def load_model():
    """Load the trained TensorFlow model"""
    global model
    try:
        # Add debugging information
        import os
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        
        # List all files in current directory
        files = os.listdir('.')
        logger.info(f"Files in current directory: {files}")
        
        model_path = './improved_mobilenet_final.keras'
        logger.info(f"Looking for model at: {model_path}")
        
        # Check if file exists
        if os.path.exists(model_path):
            logger.info("Model file found, attempting to load...")
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
            return True
        else:
            logger.error(f"Model file not found at: {model_path}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction - CORRECTED VERSION"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to model input size
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # CORRECTED: Use MobileNetV2 preprocessing
        # This scales to [-1, 1] instead of [0, 1]
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing image")
    
def predict_with_model(img_array: np.ndarray) -> Dict[str, Any]:
    """Make prediction using the loaded model"""
    global model
    
    # if model is None:
    #     # Mock prediction for demonstration
    #     return mock_prediction()
    
    try:
        # Make prediction
        predictions = model.predict(img_array)
        
        # Get the class with highest probability
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'all_predictions': {
                CLASS_NAMES[i]: float(predictions[0][i]) 
                for i in range(len(CLASS_NAMES))
            }
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error making prediction")

def mock_prediction() -> Dict[str, Any]:
    """Mock prediction for demonstration purposes"""
    import random
    
    # Generate random predictions that sum to 1
    predictions = [random.random() for _ in CLASS_NAMES]
    total = sum(predictions)
    predictions = [p / total for p in predictions]
    
    # Get highest prediction
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    return {
        'class': predicted_class,
        'confidence': confidence,
        'all_predictions': {
            CLASS_NAMES[i]: predictions[i] 
            for i in range(len(CLASS_NAMES))
        }
    }

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("=== APPLICATION STARTING UP ===")
    
    # Force immediate logging
    import sys
    sys.stdout.flush()
    
    success = load_model()
    if success:
        logger.info("=== MODEL LOADED SUCCESSFULLY ===")
    else:
        logger.error("=== MODEL LOADING FAILED ===")
    
    sys.stdout.flush()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Garbage Classification API",
        "status": "running",
        "model_loaded": model is not None,
        "supported_classes": CLASS_NAMES
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "mock",
        "timestamp": tf.timestamp().numpy().item()
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify uploaded image and return prediction with recommendations
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an image file."
        )
    
    # Check file size (limit to 10MB)
    if hasattr(file, 'size') and file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Please upload an image smaller than 10MB."
        )
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        logger.info(f"Processing image: {file.filename}, Size: {image.size}, Mode: {image.mode}")
        
        # Preprocess the image
        img_array = preprocess_image(image)
        
        # Make prediction
        prediction_result = predict_with_model(img_array)
        
        # Get recommendations for the predicted class
        recommendations = RECOMMENDATIONS.get(
            prediction_result['class'], 
            ["General recycling guidelines apply"]
        )
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "prediction": {
                "class": prediction_result['class'],
                "confidence": prediction_result['confidence'],
                "confidence_percentage": round(prediction_result['confidence'] * 100, 2)
            },
            "all_predictions": prediction_result['all_predictions'],
            "recommendations": recommendations,
            "recycling_info": {
                "category": prediction_result['class'].title(),
                "recyclable": True,
                "instructions": recommendations[:3]  # Top 3 recommendations
            }
        }
        
        logger.info(f"Prediction successful: {prediction_result['class']} ({prediction_result['confidence']:.2f})")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/classes")
async def get_classes():
    """Get list of supported classification classes"""
    return {
        "classes": CLASS_NAMES,
        "total_classes": len(CLASS_NAMES),
        "recommendations_available": list(RECOMMENDATIONS.keys())
    }

@app.get("/recommendations/{class_name}")
async def get_recommendations(class_name: str):
    """Get recycling recommendations for a specific class"""
    class_name = class_name.lower()
    if class_name not in RECOMMENDATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Recommendations not found for class: {class_name}"
        )
    
    return {
        "class": class_name,
        "recommendations": RECOMMENDATIONS[class_name]
    }

@app.get("/test-model")
async def test_model_loading():
    """Test endpoint to manually trigger model loading"""
    import os
    
    current_dir = os.getcwd()
    files = os.listdir('.')
    model_path = './improved_mobilenet_final.keras'
    model_exists = os.path.exists(model_path)
    
    if model_exists:
        try:
            test_model = tf.keras.models.load_model(model_path)
            model_info = {
                "input_shape": str(test_model.input_shape),
                "output_shape": str(test_model.output_shape),
                "layers": len(test_model.layers)
            }
            return {
                "status": "success",
                "current_dir": current_dir,
                "files": files,
                "model_exists": model_exists,
                "model_info": model_info
            }
        except Exception as e:
            return {
                "status": "error",
                "current_dir": current_dir,
                "files": files,
                "model_exists": model_exists,
                "error": str(e)
            }
    else:
        return {
            "status": "file_not_found",
            "current_dir": current_dir,
            "files": files,
            "model_exists": model_exists,
            "looking_for": model_path
        }

# For local development
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )