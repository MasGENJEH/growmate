"""
Plant Disease Classification API

This FastAPI application provides an API for classifying plant diseases using a Keras model.
It includes endpoints for health checks and image prediction with proper error handling.
"""

import os
import io
import traceback
import logging
from typing import List, Tuple, Dict, Any, Optional
import mimetypes
import imghdr

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "keras_model"
)
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
DEFAULT_INPUT_SIZE = (224, 224)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

app = FastAPI(
    title="Plant Disease Classification API",
    description="API for classifying plant diseases using a Keras model."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
input_size = DEFAULT_INPUT_SIZE

MAX_IMAGE_SIZE = 10 * 1024 * 1024

SUPPORTED_FORMATS = {'image/jpeg', 'image/png', 'image/jpg'}

class ModelService:
    """Service for model operations"""
    
    @staticmethod
    def load_model() -> Optional[tf.keras.Model]:
        """
        Load the Keras model from the specified path.
        
        Returns:
            Loaded Keras model or None if loading fails
        """
        global model, input_size
        
        try:
            if model is None:
                logger.info(f"Loading model from: {MODEL_PATH}")
                model = load_model(MODEL_PATH, compile=False)
                logger.info("Model loaded successfully")
                model.summary()
                
                input_size = ModelService.detect_input_size(model)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def detect_input_size(model: tf.keras.Model) -> Tuple[int, int]:
        """
        Detect the expected input size from the model.
        
        Args:
            model: Loaded Keras model
            
        Returns:
            Tuple of (height, width) for the model's expected input
        """
        try:
            input_shape = model.input_shape
            if input_shape and len(input_shape) == 4:
                height, width = input_shape[1], input_shape[2]
                if height is not None and width is not None:
                    logger.info(f"Detected model input size: ({height}, {width})")
                    return (height, width)
        except Exception as e:
            logger.warning(f"Could not detect model input size: {str(e)}")
        
        logger.info(f"Using default input size: {DEFAULT_INPUT_SIZE}")
        return DEFAULT_INPUT_SIZE

class ImageValidator:
    """Image validation service"""
    
    @staticmethod
    def validate_image_size(file_size: int) -> None:
        """
        Validate image file size
        
        Args:
            file_size: Size of the file in bytes
            
        Raises:
            ValueError: If file size exceeds the maximum allowed size
        """
        if file_size > MAX_IMAGE_SIZE:
            max_size_mb = MAX_IMAGE_SIZE / (1024 * 1024)
            raise ValueError(f"Image size exceeds the maximum allowed size of {max_size_mb}MB")
    
    @staticmethod
    def validate_mime_type(content_type: str) -> None:
        """
        Validate image MIME type
        
        Args:
            content_type: MIME type of the file
            
        Raises:
            ValueError: If file has an unsupported MIME type
        """
        if not content_type.startswith('image/'):
            raise ValueError("Uploaded file is not an image")
        
        if content_type not in SUPPORTED_FORMATS:
            supported_formats = ', '.join(fmt.replace('image/', '') for fmt in SUPPORTED_FORMATS)
            raise ValueError(f"Unsupported image format. Supported formats: {supported_formats}")
    
    @staticmethod
    def validate_image_file(file_bytes: bytes) -> str:
        """
        Validate image file contents
        
        Args:
            file_bytes: Image file bytes
            
        Returns:
            Image format detected
            
        Raises:
            ValueError: If file is not a valid image
        """
        if not file_bytes:
            raise ValueError("Uploaded file is empty")
        
        image_format = imghdr.what(None, file_bytes)
        if not image_format:
            raise ValueError("Uploaded file is not a valid image")
        
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img.verify()
                return image_format
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")

class ImageProcessor:
    """Service for image processing operations"""
    
    @staticmethod
    def read_image(file_bytes: bytes) -> Image.Image:
        """
        Read image file bytes and convert to PIL Image.
        
        Args:
            file_bytes: Bytes of the image file
            
        Returns:
            PIL Image object in RGB format
        
        Raises:
            ValueError: If image cannot be read or processed
        """
        try:
            return Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except UnidentifiedImageError as e:
            logger.error(f"Image format not recognized: {str(e)}")
            raise ValueError(f"Image format not recognized: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            raise ValueError(f"Error reading image: {str(e)}")
    
    @staticmethod
    def preprocess_image(img: Image.Image, target_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        Preprocess image for model prediction using multiple methods.
        
        Args:
            img: PIL Image to preprocess
            target_size: Target size (height, width) for resizing
            
        Returns:
            List of preprocessed image arrays using different methods
        
        Raises:
            ValueError: If image cannot be preprocessed
        """
        try:
            img = img.resize(target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            normalized = img_array / 255.0
            mobilenet_preprocessed = preprocess_input(img_array.copy())
            
            return [normalized, mobilenet_preprocessed]
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Error preprocessing image: {str(e)}")

class PredictionService:
    """Service for prediction operations"""
    
    @staticmethod
    def format_class_name(class_name: str) -> str:
        """
        Format class name by replacing underscores with spaces and improving readability.
        
        Args:
            class_name: Original class name with underscores
            
        Returns:
            Formatted class name with spaces
        """
        parts = class_name.split('___')
        if len(parts) == 2:
            plant, condition = parts
            plant = plant.replace('_', ' ')
            condition = condition.replace('_', ' ')
            return f"{plant} - {condition}"
        return class_name.replace('_', ' ')
    
    @staticmethod
    def predict(img_array: np.ndarray, method_idx: int) -> Dict[str, Any]:
        """
        Make a prediction using the model.
        
        Args:
            img_array: Preprocessed image array
            method_idx: Index of preprocessing method used
            
        Returns:
            Dictionary with prediction results
            
        Raises:
            ValueError: If prediction fails
        """
        loaded_model = ModelService.load_model()
        if loaded_model is None:
            raise ValueError("Model failed to load")
        
        try:
            logger.info(f"Making prediction with method #{method_idx+1}, shape: {img_array.shape}")
            
            preds = loaded_model.predict(img_array)
            pred_idx = np.argmax(preds[0])
            
            raw_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
            formatted_class = PredictionService.format_class_name(raw_class)
            
            confidence = float(np.max(preds[0]))
            
            logger.info(f"Prediction successful with method #{method_idx+1}")
            
            return {
                "prediction": formatted_class, 
                "raw_class": raw_class,
                "confidence": confidence,
                "preprocessing_method": method_idx+1
            }
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise ValueError(f"Error making prediction: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize model when application starts"""
    try:
        ModelService.load_model()
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "message": "Validation error"}),
    )

@app.get("/")
def root() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Dictionary with API status message
    """
    loaded_model = ModelService.load_model()
    if loaded_model is None:
        return {"message": "WARNING: Model failed to load. API is not fully functional."}
    return {"message": "Plant Disease Classification API is running."}

@app.post("/predict-disease")
async def predict(
    file: UploadFile = File(...),
    validate_image: bool = Query(True, description="Whether to perform comprehensive image validation")
) -> JSONResponse:
    """
    Predict plant disease from an uploaded image.
    
    Args:
        file: Uploaded image file
        validate_image: Whether to perform comprehensive image validation
        
    Returns:
        JSONResponse with prediction results
        
    Raises:
        HTTPException: For various error conditions
    """
    loaded_model = ModelService.load_model()
    if loaded_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model failed to load. Please check server logs."
        )
    
    try:
        ImageValidator.validate_mime_type(file.content_type)
        
        file_bytes = await file.read()
        
        ImageValidator.validate_image_size(len(file_bytes))
        
        if validate_image:
            image_format = ImageValidator.validate_image_file(file_bytes)
            logger.info(f"Image validated successfully, format: {image_format}")
        
        img = ImageProcessor.read_image(file_bytes)
        preprocessed_images = ImageProcessor.preprocess_image(img, input_size)
        
        last_error = None
        for i, img_array in enumerate(preprocessed_images):
            try:
                prediction_result = PredictionService.predict(img_array, i)
                return JSONResponse(content=prediction_result)
            except Exception as e:
                last_error = str(e)
                logger.error(f"Preprocessing method #{i+1} failed: {last_error}")
                continue
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"All preprocessing methods failed. Last error: {last_error}"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Prediction failed: {str(e)}"
        ) 