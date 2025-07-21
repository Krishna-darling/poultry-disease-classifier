# Reference Report for Poultry Disease Image Classification System

This report compiles and organizes references for developing a poultry disease image classification system using deep learning, focusing on Coccidiosis, Salmonella, Newcastle Disease, and Healthy states. The references are categorized to support core research, dataset acquisition, model building, deployment, and veterinary knowledge for actionable recommendations.

## A. Core Research on Poultry Disease Image Classification

1. **Poultry Diseases Diagnostics Models Using Deep Learning**  
   - **Source**: Frontiers in Artificial Intelligence  
   - **Description**: Presents an end-to-end pipeline for classifying Coccidiosis, Salmonella, Newcastle Disease, and Healthy states using fecal images. Compares convolutional neural networks (CNNs), including MobileNetV2, and evaluates their performance. Discusses field deployment and smartphone compatibility, making it critical for model selection and dataset strategy.  
   - **Utility**: Guides the selection of CNN architectures and informs dataset preparation for real-world deployment.  
   - **Link**: [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence)

2. **Poultry Disease Detection App “Chicken AI”**  
   - **Source**: GitHub  
   - **Description**: A mobile application built with Flutter and TensorFlow Lite for detecting poultry diseases from fecal images. Demonstrates TensorFlow Lite model conversion, packaging, and offline inference patterns, serving as a practical template for mobile deployment.  
   - **Utility**: Provides a reference for implementing a mobile-based diagnostic tool with offline capabilities.  
   - **Link**: [GitHub](https://github.com)

3. **Chicken Disease Image Classification**  
   - **Source**: GitHub  
   - **Description**: A transfer learning workflow using MobileNetV3Small for fecal-image-based poultry disease recognition. Includes details on class balancing, data augmentation, training, and evaluation. Offers practical notebook examples for model training.  
   - **Utility**: Serves as a hands-on guide for implementing transfer learning and optimizing training pipelines.  
   - **Link**: [GitHub](https://github.com)

## B. Public Datasets for the 4 Target Classes

1. **Poultry Diseases Detection Dataset**  
   - **Source**: Kaggle  
   - **Description**: A dataset organized into four folders (Coccidiosis, Salmonella, Newcastle, Healthy) containing JPEG images suitable for directory-based data loaders. Acts as a starter dataset for model training.  
   - **Utility**: Provides readily accessible data for initial model development and testing.  
   - **Link**: [Kaggle](https://www.kaggle.com)

2. **Supplementary Field and Lab Collections**  
   - **Source**: Frontiers (Zenodo links)  
   - **Description**: The Frontiers paper references additional field and lab-collected datasets available via Zenodo, recommended for scaling the dataset beyond the Kaggle starter set.  
   - **Utility**: Enables dataset expansion for improved model robustness and generalization.  
   - **Link**: [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence)

## C. Transfer Learning & Fine-Tuning Tutorials (Model Building)

1. **Transfer Learning & Fine-Tuning**  
   - **Source**: TensorFlow Core  
   - **Description**: Explains the concepts of feature extraction versus fine-tuning, including freezing/unfreezing layers in a Keras workflow. Guides adaptation of pretrained CNNs (e.g., MobileNet, Inception, Xception) for the four poultry disease classes.  
   - **Utility**: Essential for customizing pretrained models to specific classification tasks.  
   - **Link**: [TensorFlow](https://www.tensorflow.org)

2. **Transfer Learning with TensorFlow Hub**  
   - **Source**: TensorFlow  
   - **Description**: Demonstrates how to use pretrained ImageNet models (e.g., MobileNetV2, Inception) from TensorFlow Hub and retrain top layers for custom classes. Simplifies experimentation with different model backbones.  
   - **Utility**: Streamlines the process of testing and selecting pretrained models.  
   - **Link**: [TensorFlow](https://www.tensorflow.org/hub)

3. **Image Classification**  
   - **Source**: TensorFlow Core  
   - **Description**: Provides an end-to-end pipeline for image classification, including training and TensorFlow Lite conversion for mobile deployment.  
   - **Utility**: Supports the development of a complete workflow from training to mobile inference.  
   - **Link**: [TensorFlow](https://www.tensorflow.org)

## D. Deploying ML Models on the Web (Flask Patterns)

1. **Model Deployment: Image Classification Model Using Flask**  
   - **Source**: Analytics Vidhya  
   - **Description**: Details a Flask project structure, including template folders, static assets, model loading, and prediction routing patterns. Suitable for adapting the poultry disease classifier for web-based access.  
   - **Utility**: Offers a blueprint for building a web interface for model predictions.  
   - **Link**: [Analytics Vidhya](https://www.analyticsvidhya.com)

2. **Deploying a Keras Model as an API Using Flask**  
   - **Source**: Towards AI  
   - **Description**: Demonstrates GET/POST routes, form handling, templating, and serving predictions via web and API endpoints. Shows how to return classification results to a browser.  
   - **Utility**: Provides a framework for creating web and API-based prediction services.  
   - **Link**: [Towards AI](https://towardsai.net)

## E. Mobile / On-Device Inference (Flutter + TensorFlow Lite)

1. **TensorFlow Lite Tutorial for Flutter: Image Classification**  
   - **Source**: Kodeco  
   - **Description**: A step-by-step guide for integrating TensorFlow Lite models into Flutter applications, including image selection from camera/gallery and local inference.  
   - **Utility**: Enables adaptation of the poultry disease model for mobile applications.  
   - **Link**: [Kodeco](https://www.kodeco.com)

2. **Poultry Disease Detection App “Chicken AI”**  
   - **Source**: GitHub  
   - **Description**: Illustrates TensorFlow Lite delegate packaging across device architectures and an offline diagnosis workflow tailored for farmers.  
   - **Utility**: Serves as a reference for robust mobile deployment and offline functionality.  
   - **Link**: [GitHub](https://github.com)

## F. Veterinary Disease Knowledge (for Recommendations Screen)

1. **Coccidiosis in Poultry**  
   - **Source**: Merck Veterinary Manual  
   - **Description**: Covers etiology (Eimeria spp.), clinical signs (diarrhea, weight loss, reduced production), and control measures (anticoccidials, vaccination, management practices).  
   - **Utility**: Provides authoritative content for the app’s disease information and management recommendations.  
   - **Link**: [Merck Veterinary Manual](https://www.merckvetmanual.com)

2. **Newcastle Disease**  
   - **Source**: World Organisation for Animal Health (WOAH)  
   - **Description**: Details the highly contagious viral disease, including respiratory, neurologic, and diarrheal forms, its reportable status, lab confirmation needs, transmission, and biosecurity measures.  
   - **Utility**: Supplies critical information for farmer education and biosecurity recommendations.  
   - **Link**: [WOAH](https://www.woah.org)

3. **Salmonellosis / Bacterial Diseases in Poultry**  
   - **Source**: FAO Meat Inspection Manual  
   - **Description**: Describes clinical and postmortem findings for Salmonellosis, pullorum, fowl typhoid, and paratyphoid. Includes condemnation guidance and differential diagnosis.  
   - **Utility**: Useful for explaining disease severity and hygiene responses in the app.  
   - **Link**: [FAO](https://www.fao.org)

4. **Salmonella Framework for Raw Poultry Products**  
   - **Source**: Federal Register, April 25, 2025  
   - **Description**: Outlines regulatory efforts to reduce Salmonella contamination across the poultry production chain, emphasizing monitoring and process control.  
   - **Utility**: Provides context for educating farmers on food safety and regulatory compliance.  
   - **Link**: [Federal Register](https://www.federalregister.gov)

5. **USDA’s New Regulations to Combat Salmonella**  
   - **Source**: Poultry Producer  
   - **Description**: A plain-language summary of biosecurity, pre-harvest interventions, and HACCP-focused measures to reduce Salmonella.  
   - **Utility**: Offers farmer-friendly text for guidance on Salmonella management.  
   - **Link**: [Poultry Producer](https://www.poultryproducer.com)