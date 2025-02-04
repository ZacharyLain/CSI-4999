# general and data handling
import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# grad-cam (https://arxiv.org/abs/1610.02391, https://github.com/jacobgil/pytorch-grad-cam)
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import (show_cam_on_image, preprocess_image, deprocess_image)

def load_data(train_data_path, test_data_path, training_split):
    # Create an ImageDataGenerator for train and validation, with validation split
    datagen = ImageDataGenerator(
        rescale=1./255,  # Rescales images to 244x244
        validation_split=training_split  # Reserve 20% of the data for validation
    )

    # Training generator
    train_generator = datagen.flow_from_directory(
        directory=str(train_data_path),
        target_size=(224, 224),  # Image size that matches your model input
        batch_size=128,
        class_mode='binary',  # For binary classification: AI_GENERATED or NON_AI_GENERATED
        subset='training',  # Specify 'training' subset
        shuffle=True  # Shuffle the data
    )

    # Validation generator
    val_generator = datagen.flow_from_directory(
        directory=str(train_data_path),
        target_size=(224, 224),
        batch_size=128,
        class_mode='binary',  # Binary classification
        subset='validation',  # Specify 'validation' subset
        shuffle=False  # Don't shuffle to keep validation consistent
    )
    
    return train_generator, val_generator

def compute_weights(train_generator, val_generator):
    # Get class indices from the train generator
    class_indices = train_generator.class_indices
    print("Class indices:", class_indices)

    # Get the total number of samples in each class
    class_counts = np.bincount(train_generator.classes)
    print("Class counts:", class_counts)

    # Get the class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )

    # Create a dictionary for class weights
    class_weights_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weights_dict)
    
    return class_weights_dict

# model = Sequential([
    #     # Define the input shape
    #     tf.keras.layers.InputLayer(input_shape=input_shape),
        
    #     # 3 layer CNN
    #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
        
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
        
    #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
        
    #     # Flatten the CNN output so that we can connect it with a dense layer
    #     tf.keras.layers.Flatten(),
        
    #     # Dense layer
    #     tf.keras.layers.Dense(128, activation='relu'),
        
    #     # Output layer
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])

def create_model(input_shape, learning_rate, layers=4, fine_tune_at=140):
    # Load base resnet50 model
    base_model = resnet50.ResNet50(input_shape=input_shape,
                                   include_top=False,
                                   weights='imagenet')
    
    # Add extra convolutional layers to the base model
    x = base_model.output
    for _ in range(layers):
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    
    # Global pooling and fully connected layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.summary()
    
    # Only freeze the first fine_tune_at layers
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, class_weights, epochs, batch_size, patience):
    # Define callbacks
    # Looking to minimize validation loss
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience),
        ModelCheckpoint(monitor='val_loss', filepath=os.path.join(model_path, model_name), save_best_only=True)
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Save the model
    print(f'Saving model to {os.path.join(model_path, model_name)}...')
    model.save(os.path.join(model_path, model_name))
    
    # Return the trained model
    return model, history
    
def evaluate_model(model, history, val_generator):
    # Evaluate the model
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(val_generator, verbose=2)
    print(f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_acc:.4f}")

    cm = create_confusion_matrix_plot(model, val_generator)

    return plt, cm

def create_confusion_matrix_plot(model, val_generator):
    # 1) Generate predictions (probabilities)
    predictions = model.predict(val_generator)
    
    # 2) Binarize probabilities at 0.5
    predictions_bin = np.where(predictions > 0.5, 1, 0)
    
    # 3) True labels (order is the same as generator outputs)
    true_labels = val_generator.classes
    
    # 4) Compute confusion matrix via scikit-learn
    cm = confusion_matrix(true_labels, predictions_bin)

    # 5) Print classification report
    print("Classification Report")
    print(classification_report(true_labels, predictions_bin))

    # 6) Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["AI_GENERATED", "NON_AI_GENERATED"],
                yticklabels=["AI_GENERATED", "NON_AI_GENERATED"])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()

    return cm
    
def gradcam_heatmap(image_path, output_name, model, target_layer, target_class=None):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to('CPU')
    
    cam_algorithm = GradCAM
    with cam_algorithm(model=model,
                       target_layers=target_layer) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=target_class,
                            aug_smooth=True,
                            eigen_smooth=True)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        
        gb_model = GuidedBackpropReLUModel(model=model, device='cuda')
        gb = gb_model(input_tensor, target_category=None)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        os.makedirs(os.path.join(os.path.abspath(os.path.curdir), 'AI-Image-Detection-CNN/GradCam'), exist_ok=True)

        cam_output_path = os.path.join(os.path.join(os.path.abspath(os.path.curdir), 'AI-Image-Detection-CNN/GradCam'), f'{output_name}_cam.jpg')
        gb_output_path = os.path.join(os.path.join(os.path.abspath(os.path.curdir), 'AI-Image-Detection-CNN/GradCam'), f'{output_name}_gb.jpg')
        cam_gb_output_path = os.path.join(os.path.join(os.path.abspath(os.path.curdir), 'AI-Image-Detection-CNN/GradCam'), f'{output_name}_cam_gb.jpg')

        cv2.imwrite(cam_output_path, cam_image)
        cv2.imwrite(gb_output_path, gb)
        cv2.imwrite(cam_gb_output_path, cam_gb)
        
if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("Usage: python ai_art_classification.py <ai_art_classifcation_top_dir> <ModelOutputDir> <ModelName>")
        sys.exit(1)
    
    print()
    print(f'##########\nTensorFlow\n##########')
    print(f'Version: {tf.__version__}')               # Check TensorFlow version
    print(f'GPU: {tf.config.list_physical_devices("GPU")}')  # Check if GPU is available
    print(f'Cuda Available: {tf.test.is_built_with_cuda()}') # Check if TensorFlow was built with CUDA support
    
    print(f'\n##########\nTorch\n##########')
    print(f'Version: {torch.__version__}')            # Check PyTorch version
    print(f'Cuda Version:{torch.version.cuda}')           # Check which CUDA version PyTorch was built against
    print(f'Cuda Available: {torch.cuda.is_available()}')    # Should be True if everything worked
    print()
    
    try:
        # Check if the data directory exists
        if not os.path.exists(sys.argv[1]):
            print('Error: The data directory does not exist.')
            print('Make sure the argument does not include a slash at the beginning.')
            sys.exit(1)
    except Exception as e:
        sys.exit(1)
        
    current_dir = os.path.abspath(os.path.curdir)
         
    # Define paths to the AI-generated and real image data sources
    train_data_path = os.path.join(os.path.abspath(os.path.curdir), sys.argv[1], 'train')
    test_data_path = os.path.join(os.path.abspath(os.path.curdir), sys.argv[1], 'test')
    
    print('Training Data Path:', train_data_path)
    print('Testing Data Path:', test_data_path)

    # Load the data
    print('Loading data...')
    train_generator, val_generator = load_data(train_data_path, test_data_path, 0.2)
    
    # Compute class weights
    print('Computing class weights...')
    class_weights = compute_weights(train_generator, val_generator)
    
    # Define the input shape
    input_shape = (224, 224, 3)
    
    # Define the model
    print('Creating model...')
    model_path = os.path.join(current_dir, sys.argv[2], 'model')
    model_name = f'{sys.argv[3]}_model.h5'
    model = create_model(input_shape, 0.0001, 5)
    
    # Train the model
    print('Training model...')
    trained_model, model_history = train_model(model, class_weights, 50, 32, 5)
    
    # Evaluate the model
    print('Evaluating model...')
    plt, cm = evaluate_model(trained_model, model_history, val_generator)

    # Visalize the model evaluation
    plt.show()
    
    #
    # Create a GradCAM explainer
    #
    target_layer = model.get_layer('conv2d_4')
    
    gradcam_heatmap(os.path.join(current_dir, 'RawData/ai_art_classification/train/AI_GENERATED/0.jpg'),
                    'real_img_1', model, target_layer, target_class=None)
    gradcam_heatmap(os.path.join(current_dir, 'RawData/ai_art_classification/train/NON_AI_GENERATED/3.jpg'),
                    'fake_img_1', model, target_layer, target_class=None)